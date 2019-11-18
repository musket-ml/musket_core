import os
import sys

from musket_core.datasets import PredictionItem, inherit_dataset_params, CompositeDataSet,DataSet
from musket_core.utils import load,save,readArray,dumpArray
from musket_core import context
import tqdm
import numpy as np
import threading
from musket_core import utils

__lock__ = threading.Lock()
storage = {}

class Cache(DataSet):

    def __init__(self,parent):
        self.parent=parent
        self.k={}
        if hasattr(parent,"folds"):
            self.folds=getattr(parent,"folds");


    def __getitem__(self, item):
        if item in self.k:
            return self.k[item]

        v=self.parent[item]
        self.k[item]=v
        return v

    def __len__(self):
        return len(self.parent)
    
    


class CachedPredictionItem(PredictionItem):

    def __init__(self, path, x, y,originalDataSet):
        super().__init__(path,x,y)
        self._original=originalDataSet

    def original(self):
        return self._original[self.id]

    def rootItem(self):
        return self.original().rootItem()

    def item_id(self):
        return self.original().item_id()

class DiskCache(DataSet):

    def __init__(self,parent,items):
        self.parent=parent
        self.items=items
        inherit_dataset_params(parent, self)

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = list(item)
            result = [self.__getitem__(i) for i in indices]
            return result
        else:
            isList = isinstance(self.items[0],list)
            if isList:
                return CachedPredictionItem(item,list(map(lambda x: x[item], self.items[0])),self.items[1][item],self.parent)
            else:
                return CachedPredictionItem(item,self.items[0][item],self.items[1][item],self.parent)

    def __len__(self):
        return len(self.parent)

class DiskCache1(DataSet):

    def __init__(self,parent,items,xIsListOrTuple,yIsListOrTuple):
        self.parent=parent
        self.items=items
        self.xIsListOrTuple = xIsListOrTuple
        self.yIsListOrTuple = yIsListOrTuple
        inherit_dataset_params(parent, self)
        if hasattr(parent, "name"):
            self.origName = getattr(parent,"name");


    def __getitem__(self, item):
        if isinstance(item,slice):
            indices = list(item)
            result = [self.__getitem__(i) for i in indices]
            return result
        else:
            x = self.items[0][item] if not self.xIsListOrTuple else [self.items[0][c][item] for c in range(len(self.items[0]))]
            y = self.items[1][item] if not self.yIsListOrTuple else [self.items[1][c][item] for c in range(len(self.items[1]))]
            return CachedPredictionItem(item,x,y,self.parent)

    def __len__(self):
        return len(self.parent)

def cache(layers,declarations,config,outputs,linputs,pName,withArgs):
    def ccc(input):
        return Cache(input)

    return ccc

CACHE_DIR=None

def get_cache_dir():
    if CACHE_DIR is not None:
        return CACHE_DIR
    cp=context.get_current_project_path()
    if cp is None:
        cp=os.getcwd()
    d=os.path.join(cp,".cache/")
    utils.ensure(d)
    return d

def cache_name(input:DataSet):
    name = "data"
    id = "dataset"
    if hasattr(input, "name"):
        id = getattr(input, "name")
        name = id.replace("{", "").replace("[", "").replace("/", "").replace("\\", "").replace("]", "").replace("}", "").replace(" ", "").replace(",", "").replace("\'", "").replace(":", "")
    return name

def diskcache_new(layers,declarations,config,outputs,linputs,pName,withArgs):
    def ccc(input):

        __lock__.acquire()
        try:
            return ccc1(input)
        finally:
            __lock__.release()




    def ccc1(input):
        

        try:
            name = cache_name(input)
            name=get_cache_dir()+name
            if name in storage:
                return storage[name]

            if isinstance(input, CompositeDataSet):
                components = list(map(lambda x:ccc1(x), input.components))
                compositeDS = CompositeDataSet(components)
                inherit_dataset_params(input, compositeDS)
                if hasattr(input, "name"):
                    compositeDS.origName = input.name
                return compositeDS

            data = None
            xStructPath = f"{name}/x.struct"
            yStructPath = f"{name}/y.struct"
            blocksCountPath = f"{name}/blocks_count.int"
            if os.path.exists(xStructPath) and os.path.exists(yStructPath) and os.path.exists(blocksCountPath):
                blocksCount = load(blocksCountPath)
                xStruct = load(xStructPath)
                yStruct = load(yStructPath)
                xIsListOrTuple = xStruct[2] in ["list", "tuple"]
                yIsListOrTuple = yStruct[2] in ["list", "tuple"]

                xData, yData = init_buffers(xStruct, yStruct)

                for blockInd in tqdm.tqdm(range(blocksCount), "loading disk cache for:" + name):
                    if not xIsListOrTuple:
                        blockPath = f"{name}/x_{blockInd}.dscache"
                        if os.path.exists(blockPath):
                            xBuff = load(blockPath)
                            for x in xBuff:
                                xData.append(x)
                        else:
                            raise Exception(f"Cache block is missing: {name}")
                    else:
                        for c in range(len(xStruct[0])):
                            blockPath = f"{name}/x_{blockInd}_{c}.dscache"
                            if os.path.exists(blockPath):
                                xBuff = load(blockPath)
                                for x in xBuff:
                                    xData[c].append(x)
                            else:
                                raise Exception(f"Cache block is missing: {name}")
                    if not yIsListOrTuple:
                        blockPath = f"{name}/y_{blockInd}.dscache"
                        if os.path.exists(blockPath):
                            yBuff = load(blockPath)
                            for y in yBuff:
                                yData.append(y)
                        else:
                            raise Exception(f"Cache block is missing: {name}")
                    else:
                        for c in range(len(yStruct[0])):
                            blockPath = f"{name}/y_{blockInd}_{c}.dscache"
                            if os.path.exists(blockPath):
                                yBuff = load(blockPath)
                                for y in yBuff:
                                    yData[c].append(y)
                            else:
                                raise Exception(f"Cache block is missing: {name}")

                data = (xData, yData)

            if data is None:
                if not os.path.isdir(name):
                    os.mkdir(name)

                i0 = input[0]
                i0x = i0.x
                i0y = i0.y
                l = len(input)

                xStruct = inspect_structure(i0x)
                yStruct = inspect_structure(i0y)

                xIsListOrTuple = xStruct[2] in ["list", "tuple"]
                yIsListOrTuple = yStruct[2] in ["list", "tuple"]

                xData, yData = init_buffers(xStruct, yStruct)

                buffSize = 0

                barrier = 64 * 1024 * 1024

                blockInd = 0
                for i in tqdm.tqdm(range(l), "building disk cache for:" + id):
                    item = input[i]
                    if not xIsListOrTuple:
                        xData.append(item.x)
                    else:
                        for c in range(len(xStruct[0])):
                            xData[c].append(item.x[c])
                    if not yIsListOrTuple:
                        yData.append(item.y)
                    else:
                        for c in range(len(yStruct[0])):
                            yData[c].append(item.y[c])

                    buffSize += get_size(item.x)
                    buffSize += get_size(item.y)

                    if buffSize > barrier or i == l-1:

                        if not xIsListOrTuple:
                            arr = xData
                            if xStruct[0][0].startswith("int") or xStruct[0][0].startswith("float"):
                                arr = np.array(arr)
                            save(f"{name}/x_{blockInd}.dscache", arr)
                        else:
                            for c in range(len(xStruct[0])):
                                arr = xData[c]
                                if xStruct[0][c].startswith("int") or xStruct[0][c].startswith("float"):
                                    arr = np.array(arr)
                                save(f"{name}/x_{blockInd}_{c}.dscache", arr)

                        if not yIsListOrTuple:
                            arr = yData
                            if yStruct[0][0].startswith("int") or yStruct[0][0].startswith("float"):
                                arr = np.array(arr)
                            save(f"{name}/y_{blockInd}.dscache", arr)
                        else:
                            for c in range(len(yStruct[0])):
                                arr = yData[c]
                                if yStruct[0][c].startswith("int") or yStruct[0][c].startswith("float"):
                                    arr = np.array(arr)
                                save(f"{name}/y_{blockInd}_{c}.dscache", arr)

                        buffSize = 0
                        blockInd += 1
                        xData, yData = init_buffers(xStruct, yStruct)
                        pass

                save(xStructPath, xStruct)
                save(yStructPath, yStruct)
                save(blocksCountPath, blockInd)
                return ccc1(input)
            result = DiskCache1(input, data, xIsListOrTuple, yIsListOrTuple)
            storage[name] = result
            return result
        finally:
            pass

    return ccc


def get_size(obj) -> int:
    if isinstance(obj, list):
        sum = 0
        for x in obj:
            sum += get_size(x)
        return sum
    elif isinstance(obj, tuple):
        sum = 0
        for x in list(obj):
            sum += get_size(x)
        return sum
    elif isinstance(obj, np.ndarray):
        return obj.size * obj.dtype.itemsize
    else:
        return sys.getsizeof(obj)


def init_buffers(xStruct, yStruct):
    xIsListOrTuple = xStruct[2] in ["list", "tuple"]
    yIsListOrTuple = yStruct[2] in ["list", "tuple"]
    xData = []
    if xIsListOrTuple:
        for i in range(len(xStruct[0])):
            xData.append([])
    yData = []
    if yIsListOrTuple:
        for i in range(len(yStruct[0])):
            yData.append([])
    return xData, yData

def inspect_structure(obj)->([str], [[int]], str):
    container = None
    isTuple = isinstance(obj, tuple)
    if isTuple:
        container = 'tuple'
    isList = isinstance(obj, list)
    if isList:
        container = 'list'

    types = []
    shapes = []
    if isTuple:
        for ch in list(obj):
            chStruct = inspect_structure(ch)
            types.extend(chStruct[0])
            shapes.extend(chStruct[1])
    elif isList:
        for ch in obj:
            chStruct = inspect_structure(ch)
            types.extend(chStruct[0])
            shapes.extend(chStruct[1])
    elif isinstance(obj,np.ndarray):
        tstr = str(obj.dtype)
        types.append(tstr)
        shapes.append(list(obj.shape))
        pass
    elif isinstance(obj, str):
        types.append('str')
        shapes.append([])
        pass

    return types, shapes, container

def diskcache_old(layers,declarations,config,outputs,linputs,pName,withArgs):
    def ccc(input):
        try:
            __lock__.acquire()
            return ccc1(input)
        finally:
            __lock__.release()

    def ccc1(input):       
        try:
            if hasattr(context.context,"no_cache"):
                return input
            name = "data"
            id = "dataset"
            l = len(input)

            if hasattr(input, "name"):
                id = getattr(input, "name")
                name = id.replace("{", "").replace("[", "").replace("/", "").replace("\\", "").replace("]", "").replace(
                    "}", "").replace(" ", "").replace(",", "").replace("\'", "").replace(":", "")

            name=get_cache_dir()+name
            if name in storage:
                r= storage[name]
                inherit_dataset_params(input, r) 
                return r     

            if isinstance(input, CompositeDataSet):
                components = list(map(lambda x:ccc1(x), input.components))
                compositeDS = CompositeDataSet(components)
                inherit_dataset_params(input, compositeDS)
                if hasattr(input, "name"):
                    compositeDS.origName = input.name
                return compositeDS

            i0 = input[0]
            i0x = i0.x
            xIsList = isinstance(i0x,list)
            i0y = i0.y
            if not xIsList:
                shapeX = np.concatenate(([l], i0x.shape))
            else:
                shapeX = list(map(lambda x: np.concatenate(([l], x.shape)), i0x))
            
            yIsList = isinstance(i0y,list)    
            if not yIsList:
                shapeY = np.concatenate(([l], i0y.shape))
            else:
                shapeY = list(map(lambda x: np.concatenate(([l], x.shape)), i0y))    
            data = None
            ext = "dscache"
            if os.path.exists(name):
                if not os.path.isdir(name):
                    #old style
                    data = load(name)
                elif os.path.exists(f"{name}/x_0.{ext}"):
                    if not xIsList:
                        data = (np.zeros(shapeX, i0x.dtype), np.zeros(shapeY, i0y.dtype))
                    else:
                        data = (list(map(lambda x: np.zeros(x, i0x[0].dtype), shapeX)), np.zeros(shapeY, i0y.dtype))
                    try:
                        readArray(data[0], f"{name}/x_", ext, "Loading X cache...", l)
                    except ValueError:
                        raise ValueError(f"Stored X has unexpected size for dataset '{name}'. Path: " + name)

                    try:
                        readArray(data[1], f"{name}/y_", ext, "Loading Y cache...", l)
                    except ValueError:
                        raise ValueError(f"Stored Y has unexpected size for dataset '{name}'. Path: " + name)

            if data is None:
                if not xIsList:
                    data = (np.zeros(shapeX, i0x.dtype), np.zeros(shapeY, i0y.dtype))
                else:
                    data = (list(map(lambda x: np.zeros(x,i0x[0].dtype), shapeX)), np.zeros(shapeY, i0y.dtype))

                # if not xIsList:
                #     def func(i):
                #         data[0][i] = input[i].x
                #         data[1][i] = input[i].y
                # else:
                #     def func(i):
                #         for j in range(len(shapeX)):
                #             data[0][j][i] = input[i].x[j]
                #             data[1][i] = input[i].y

                # pool = Pool(4)
                # zip(*pool.map(func, range(0, l)))


                for i in tqdm.tqdm(range(l), "building disk cache for:" + id):
                    if not xIsList:
                        data[0][i] = input[i].x
                    else:
                        for j in range(len(shapeX)):
                            data[0][j][i] = input[i].x[j]
                    data[1][i] = input[i].y

                if not os.path.isdir(name):
                    os.mkdir(name)
                dumpArray(data[0], f"{name}/x_", ext, "Saving X cache...")
                dumpArray(data[1], f"{name}/y_", ext, "Saving Y cache...")

            result = DiskCache(input, data)

            inherit_dataset_params(input, result)            
            if hasattr(input, "name"):
                result.origName = input.name
            storage[name] = result
            return result
        finally:
            pass
    return ccc


def diskcache(layers, declarations, config, outputs, linputs, pName, withArgs):
    if config is not None and isinstance(config, dict):
        if 'split' in config and config['split'] == True:
            return diskcache_new(layers, declarations, config, outputs, linputs, pName, withArgs)

    return diskcache_old(layers, declarations, config, outputs, linputs, pName, withArgs)

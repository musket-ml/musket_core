import os
from musket_core.datasets import PredictionItem
from musket_core.utils import load,save,readArray,dumpArray
import tqdm
import numpy as np
import threading

__lock__ = threading.Lock()
storage = {}

class Cache:

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


class DiskCache:

    def __init__(self,parent,items):
        self.parent=parent
        self.items=items
        if hasattr(parent,"folds"):
            self.folds=getattr(parent,"folds");


    def __getitem__(self, item):
        isList = isinstance(self.items[0],list)
        if isList:
            return PredictionItem(item,list(map(lambda x: x[item], self.items[0])),self.items[1][item])
        else:
            return PredictionItem(item,self.items[0][item],self.items[1][item])

    def __len__(self):
        return len(self.parent)

def cache(layers,declarations,config,outputs,linputs,pName,withArgs):
    def ccc(input):
        return Cache(input)

    return ccc

CACHE_DIR=""
def diskcache(layers,declarations,config,outputs,linputs,pName,withArgs):
    def ccc(input):
        global CACHE_DIR
        __lock__.acquire()
        try:
            name = "data"
            id = "dataset"
            l = len(input)

            if hasattr(input, "name"):
                id = getattr(input, "name")
                name = id.replace("{", "").replace("[", "").replace("]", "").replace("}", "").replace(" ", "").replace(",","").replace("\'","").replace(":", "")
            name=CACHE_DIR+name
            if name in storage:
                return storage[name]

            i0 = input[0]
            i0x = i0.x
            xIsList = isinstance(i0x,list)
            i0y = i0.y
            if not xIsList:
                shapeX = np.concatenate(([l], i0x.shape))
            else:
                shapeX = list(map(lambda x: np.concatenate(([l], x.shape)), i0x))
            shapeY = np.concatenate(([l], i0y.shape))
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
            if hasattr(input, "name"):
                result.origName = input.name
            storage[name] = result
            return result
        finally:
            __lock__.release()
    return ccc
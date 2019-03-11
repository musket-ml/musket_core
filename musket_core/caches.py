import os
from musket_core.datasets import PredictionItem
from musket_core.utils import load,save,readArray,dumpArray
import tqdm
import numpy as np
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
        return PredictionItem(item,self.items[0][item],self.items[1][item])

    def __len__(self):
        return len(self.parent)

def cache(layers,declarations,config,outputs,linputs,pName,withArgs):
    def ccc(input):
        return Cache(input)

    return ccc

def diskcache(layers,declarations,config,outputs,linputs,pName,withArgs):
    def ccc(input):
        name = "data"
        id = "dataset"
        l = len(input)

        if hasattr(input, "name"):
            id = getattr(input, "name")
            name = id.replace("{", "").replace("[", "").replace("]", "").replace("}", "").replace(" ", "").replace(",","").replace("\'","").replace(":", "")

        i0 = input[0]
        i0x = i0.x
        i0y = i0.y
        shapeX = np.concatenate(([l], i0x.shape))
        shapeY = np.concatenate(([l], i0y.shape))
        data = None
        ext = "dscache"
        if os.path.exists(name):
            if not os.path.isdir(name):
                #old style
                data = load(name)
            elif os.path.exists(f"{name}/x_0.{ext}"):
                data = (np.zeros(shapeX, i0x.dtype), np.zeros(shapeY, i0y.dtype))
                try:
                    readArray(data[0], f"{name}/x_", ext, l)
                except ValueError:
                    raise ValueError(f"Stored X has unexpected size for dataset '{name}'. Path: " + name)

                try:
                    readArray(data[1], f"{name}/y_", ext, l)
                except ValueError:
                    raise ValueError(f"Stored Y has unexpected size for dataset '{name}'. Path: " + name)

        if data is None:
            data = (np.zeros(shapeX, i0x.dtype), np.zeros(shapeY, i0y.dtype))
            for i in tqdm.tqdm(range(l), "building disk cache for:" + id):
                data[0][i] = input[i].x
                data[1][i] = input[i].y

            if not os.path.isdir(name):
                os.mkdir(name)
            dumpArray(data[0], f"{name}/x_", ext)
            dumpArray(data[1], f"{name}/y_", ext)

        return DiskCache(input, data)
    return ccc
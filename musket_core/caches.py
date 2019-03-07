import os
from musket_core.datasets import PredictionItem
from musket_core.utils import load,save
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
        name="data"
        id="dataset"
        if hasattr(input,"name"):
            id=getattr(input,"name")
            name=id.replace("{","").replace("[","").replace("]","").replace("}","").replace(" ","").replace(",","").replace("\'","").replace(":","")
        if os.path.exists(name):
            data=load(name)
        else:
            X=[]
            Y=[]
            for i in tqdm.tqdm(range(len(input)),"building disk cache for:"+id):
                X.append(input[i].x)
                Y.append(input[i].y)
            data=(np.array(X), np.array(Y))
        save(name,data)
        return DiskCache(input,data)
    return ccc
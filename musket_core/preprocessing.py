import functools
import importlib
import inspect
from musket_core.datasets import PredictionItem

class PreprocessedDataSet:

    def __init__(self,parent,func,**kwargs):
        self.parent=parent
        self.func=func
        self.kw=kwargs

        if hasattr(parent,"folds"):
            self.folds=getattr(parent,"folds");
        if hasattr(parent, "name"):
            self.name=parent.name+self.func.__name__+str(kwargs)
            self.origName = self.name

        pass

    def __getitem__(self, item):
        pi=self.parent[item]
        x=self.func(pi.x,**self.kw)
        newPi = PredictionItem(item,x,pi.y)
        return newPi

    def __len__(self):
        return len(self.parent)



def dataset_preprocessor(func):
    def wrapper(input,**kwargs):
        return PreprocessedDataSet(input,func,**kwargs)
    wrapper.args=inspect.signature(func).parameters
    wrapper.preprocessor=True
    wrapper.__name__=func.__name__
    return wrapper
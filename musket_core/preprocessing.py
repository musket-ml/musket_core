import functools
import importlib
import inspect
from musket_core.datasets import PredictionItem

class PreproccedPredictionItem(PredictionItem):

    def __init__(self, path, x, y,original:PredictionItem):
        super().__init__(path,x,y)
        self._original=original

    def original(self):
        return self._original

    def rootItem(self):
        return self._original.rootItem()

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
        newPi = PreproccedPredictionItem(pi.id,x,pi.y,pi)
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
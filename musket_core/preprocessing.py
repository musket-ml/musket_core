import functools
import importlib
import inspect
from musket_core.datasets import PredictionItem,DataSet

class PreproccedPredictionItem(PredictionItem):

    def __init__(self, path, x, y,original:PredictionItem):
        super().__init__(path,x,y)
        self._original=original

    def original(self):
        return self._original

    def rootItem(self):
        return self._original.rootItem()


class PreprocessedDataSet(DataSet):

    def __init__(self,parent,func,**kwargs):
        super().__init__()
        self.parent=parent
        self.func=func
        self.kw=kwargs
        if isinstance(parent,PreprocessedDataSet) or isinstance(parent,DataSet):
            self._parent_supports_target=True
        else:
            self._parent_supports_target = False
        if hasattr(parent,"folds"):
            self.folds=getattr(parent,"folds");
        if hasattr(parent,"holdoutArr"):
            self.holdoutArr=getattr(parent,"holdoutArr");
        if hasattr(parent, "name"):
            self.name=parent.name+self.func.__name__+str(kwargs)
            self.origName = self.name
        pass

    def id(self):
        m=self.func.__name__
        if len(self.kw)>0:
            m=m+":"+str(self.kw)
        return m

    def get_target(self,item):
        if self._parent_supports_target:
            return self.parent.get_target(item)
        return self[item].y

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
    wrapper.original=func
    wrapper.__name__=func.__name__
    return wrapper
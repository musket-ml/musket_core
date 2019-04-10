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

    def __init__(self,parent,func,expectsItem,**kwargs):
        super().__init__()
        self.expectsItem = expectsItem
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
        if self._parent_supports_target and not self.expectsItem:
            return self.parent.get_target(item)
        return self[item].y

    def __getitem__(self, item):
        pi=self.parent[item]
        arg = pi if self.expectsItem else pi.x
        result = self.func(arg,**self.kw)
        newPi = result if self.expectsItem else PreproccedPredictionItem(pi.id,result,pi.y,pi)
        return newPi

    def __len__(self):
        return len(self.parent)


def dataset_preprocessor(func):
    expectsItem = False
    params = inspect.signature(func).parameters
    if 'input' in params:
        inputParam = params['input']
        if hasattr(inputParam, 'annotation'):
            pType = inputParam.annotation
            if hasattr(pType, "__module__") and hasattr(pType, "__name__"):
                if pType.__module__ == "musket_core.datasets" and pType.__name__ == "PredictionItem":
                    expectsItem = True

    def wrapper(input,**kwargs):
        return PreprocessedDataSet(input,func,expectsItem,**kwargs)
    wrapper.args=inspect.signature(func).parameters
    wrapper.preprocessor=True
    wrapper.original=func
    wrapper.__name__=func.__name__
    return wrapper
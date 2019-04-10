import functools
import importlib
import inspect
from  typing import List,Optional

from py4j.tests.java_callback_test import A

from musket_core.datasets import PredictionItem,DataSet,get_id,get_stages,get_stage


class PreproccedPredictionItem(PredictionItem):

    def __init__(self, path, x, y,original:PredictionItem):
        super().__init__(path,x,y)
        self._original=original

    def original(self):
        return self._original

    def rootItem(self):
        return self._original.rootItem()


class AbstractPreprocessedDataSet(DataSet):

    def __init__(self,parent):
        super().__init__()
        self.expectsItem = False
        self.parent=parent
        if isinstance(parent,PreprocessedDataSet) or isinstance(parent,DataSet):
            self._parent_supports_target=True
        else:
            self._parent_supports_target = False
        if hasattr(parent,"folds"):
            self.folds=getattr(parent,"folds")
        if hasattr(parent,"holdoutArr"):
            self.holdoutArr=getattr(parent,"holdoutArr")

    def __len__(self):
        return len(self.parent)

    def get_target(self,item):
        if self._parent_supports_target and not self.expectsItem:
            return self.parent.get_target(item)
        return self[item].y


class PreprocessedDataSet(AbstractPreprocessedDataSet):

    def __init__(self,parent,func,expectsItem,**kwargs):
        super().__init__(parent)
        self.expectsItem = expectsItem
        self.func=func
        self.kw=kwargs
        if hasattr(parent, "name"):
            self.name=parent.name+self.func.__name__+str(kwargs)
            self.origName = self.name
        pass

    def id(self):
        m=self.func.__name__
        if len(self.kw)>0:
            m=m+":"+str(self.kw)
        return m

    def __getitem__(self, item):
        pi=self.parent[item]
        arg = pi if self.expectsItem else pi.x
        result = self.func(arg,**self.kw)
        newPi = result if self.expectsItem else PreproccedPredictionItem(pi.id,result,pi.y,pi)
        return newPi




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


_num_splits=0


class SplitPreproccessor(AbstractPreprocessedDataSet):

    def __init__(self,parent,branches:List[PreprocessedDataSet]):
        global _num_splits
        self.num=_num_splits
        _num_splits=_num_splits+1
        super().__init__(parent)
        self.branches=branches

    def id(self):
        return "split("+str(self.num)+")"

    def subStages(self):
        res=[]
        for m in self.branches:
            res=res+get_stages(m)
        return res

    def get_stage(self,name)->Optional[DataSet]:
        for m in self.branches:
            s=get_stage(m,name)
            if s is not None:
                return s
        return None

    def __getitem__(self, item):
        items=[x[item] for x in self.branches]
        return PreproccedPredictionItem(items[0].id,tuple([x.x for x in items]),items[0].y,items[0].original())



import inspect
from  typing import List,Optional
import numpy as np

from musket_core.datasets import PredictionItem,DataSet,get_id,get_stages,get_stage,inherit_dataset_params, CompositeDataSet


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
        inherit_dataset_params(parent,self)

    def __len__(self):
        return len(self.parent)

    def get_target(self,item):
        if self._parent_supports_target and not self.expectsItem:
            return self.parent.get_target(item)
        return self[item].y
    
    def encode(self,item,encode_y=False,treshold=0.5):
        return self.parent.encode(item,encode_y,treshold)
    
    def root(self):
        if hasattr(self.parent,"root"):
            return self.parent.root()
        return self.parent
    
    def _create_dataframe(self,items):
        return self.parent._create_dataframe(items)

def deployHandler(origin=None): 
    def inner(func):         
        func.deployHandler=origin
        return func        
    return inner #this is the fun_obj mentioned in the above content

def _sorted_args(args):
    rs={}
    for k in sorted(args.keys()):
        rs[k]=args[k]
    return rs;
class PreprocessedDataSet(AbstractPreprocessedDataSet):

    def __init__(self,parent,func,expectsItem,**kwargs):
        super().__init__(parent)
        self.expectsItem = expectsItem
        self.func=func
        self.kw=kwargs
        if hasattr(parent, "name"):
            self.name=parent.name+self.func.__name__+str(_sorted_args(kwargs))
            self.origName = self.name
        pass

    def id(self):
        m=self.func.__name__
        if len(self.kw)>0:
            m=m+":"+str(self.kw)
        return m

    def __getitem__(self, item):
        pi=self.parent[item]
        isSlice = False
        if isinstance(item, slice) and isinstance(pi,list):
            isSlice= True
            piList = pi
        else:
            piList = [ pi ]

        newList = []
        for pi in piList:
            newPi = self.execute_transformation(pi)
            newList.append(newPi)

        return newList if isSlice else newList[0]

    def execute_transformation(self, pi):
        arg = pi if self.expectsItem else pi.x
        result = self.func(arg, **self.kw)
        newPi = result if self.expectsItem else PreproccedPredictionItem(pi.id, result, pi.y, pi)
        return newPi


def dataset_transformer(func):
    def wrapper(input,*args,**kwargs):
        f=func
        res=func(input,*args,**kwargs)
        if hasattr(input, "name"):
            res.name = input.name + f.__name__ + str(_sorted_args(kwargs))
            res.origName = res.name
        return res
        
    wrapper.args=inspect.signature(func).parameters
    wrapper.preprocessor=True
    wrapper.original=func
    wrapper.__name__=func.__name__
    return wrapper

def take_nth(num,parent):
    def n(item):
        return item[num]
    return PreprocessedDataSet(parent,n,False)

def dataset_preprocessor(func):
    expectsItem = False
    
        
    params = inspect.signature(func).parameters
    inputParam=None
    if 'input' in params:
        inputParam = params['input']
    if 'inp' in params:
        inputParam = params['inp']
    if inputParam is not None:    
        if hasattr(inputParam, 'annotation'):
                pType = inputParam.annotation
                if hasattr(pType, "__module__") and hasattr(pType, "__name__"):
                    if pType.__module__ == "musket_core.datasets" and pType.__name__ == "PredictionItem":
                        expectsItem = True
    
    def wrapper(input,**kwargs):
        f=func
        if inspect.isclass(f):
            n=f.__name__
            f=f(**kwargs)
            f.__name__=n
            kwargs={}  
        if isinstance(input,CompositeDataSet):
            components = list(map(lambda x: PreprocessedDataSet(x,f,expectsItem,**kwargs), input.components))
            compositeDS = CompositeDataSet(components)
            inherit_dataset_params(input, compositeDS)
            if hasattr(input, "name"):
                compositeDS.name = input.name + f.__name__ + str(_sorted_args(kwargs))
                compositeDS.origName = compositeDS.name
            return compositeDS
        else:
            return PreprocessedDataSet(input,f,expectsItem,**kwargs)
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
        if isinstance(branches[0], PreprocessedDataSet) or isinstance(branches[0], DataSet):
            self._parent_supports_target = True
        contributions=[]
        hasContributions=False    
        for x in branches:
            if hasattr(x, "contribution"):    
                hasContributions=True
                contributions.append(getattr(x, "contribution"))
            else: contributions.append(None)
        if hasContributions:
            self.contributions=contributions
        self.name="-".join([x.name for x in branches])        
                         
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
        return PreproccedPredictionItem(items[0].id,[x.x for x in items],items[0].y,items[0].original())

    def get_target(self,item):
        if self._parent_supports_target and not self.expectsItem:
            return self.branches[0].get_target(item)
        return self[item].y


class SplitConcatPreprocessor(SplitPreproccessor):

    def __init__(self,parent,branches:List[PreprocessedDataSet],axis=-1):
        super().__init__(parent,branches)
        self.axis=axis

    def __getitem__(self, item):
        items=[x[item] for x in self.branches]
        return PreproccedPredictionItem(items[0].id,np.concatenate([x.x for x in items],axis=self.axis),items[0].y,items[0].original())
    
    
    

class Augmentation(AbstractPreprocessedDataSet):

    def __init__(self,parent,seq:List[PreprocessedDataSet],weights:[float] or np.ndarray,seed:int):
        super().__init__(parent)
        self.seq = seq
        self.weights = weights if isinstance(weights,np.ndarray) else np.array(weights)
        self.seed = seed

        state = np.random.get_state()
        np.random.seed(self.seed)
        self.state = np.random.get_state()
        np.random.set_state(state)

    def id(self):
        str1 = "; ".join([f"{x[0].id}:{x[1]}" for x in zip(self.seq,self.weights.tolist())])
        result = f"augment({self.seed}; {str1})"
        return result

    def __getitem__(self, item: int or slice):
        return self.parent[item]

    def get_train_item(self, item):
        input = self.parent[item]
        isSlice = isinstance(item,slice)
        iArr = [ input ] if not isSlice else input
        oArr = []

        for x in iArr:

            activate = self.flip()
            for i in range(len(self.seq)):
                if activate[i]:
                    x = self.seq[i].execute_transformation(x)

            oArr.append(x)

        output = oArr[0] if not isSlice else oArr
        return output

    def flip(self)->np.ndarray:
        state = np.random.get_state()
        np.random.set_state(self.state)

        rand = np.random.uniform(size=len(self.weights))

        self.state = np.random.get_state()
        np.random.set_state(state)

        return rand < self.weights

from musket_core import datasets
import numpy as np
import sys
import traceback
from tqdm import tqdm

class ShapeMeta:
    def __init__(self):
        pass

def get_shape(x):
    if isinstance(x,int):
        return (1,)
    if isinstance(x,float):
        return (1,)
    if isinstance(x,bool):
        return (1,)
    if isinstance(x,str):
        return (len(x),)
    if isinstance(x,tuple):
        return tuple(get_shape(i) for i in x)
    if isinstance(x,list):
        #this is not clear situation
        try:
            return np.array(x).shape
        except:
            raise ValueError("Bad shape")
    if isinstance(x,np.ndarray):
        return x.shape


class Classification:

    pass

class BinaryClassification(Classification):

    def __init__(self):
        self.items=set()
        self.numFirst=False
        self.numSecond=False
        self.firstItem=None
        pass

    def check(self,item):
        if isinstance(item, list) or isinstance(item, np.ndarray):
            if len(item)>1:
                return False
            item=item[0]
        try:
            self.items.add(item)
            if len(self.items)>2:
                return False
            if self.firstItem is None:
                self.firstItem=item
            if item==self.firstItem:
                self.numFirst=self.numFirst+1
            else:
                self.numSecond=self.numSecond+1
            return True
        except:
            return False


class MultiClassification(Classification):

    def __init__(self):
        self.classCount=None
        self.counts=None
        pass

    def check(self,item):
        if isinstance(item, list) or isinstance(item, np.ndarray):
            _len=len(item)
            if _len==1:
                return False
            if self.classCount is None:
                self.classCount=_len
                self.counts = [dict() for x in range(_len)]
            elif self.classCount!=_len:
                return False
            for v in range(len(item)):
                z=item[v]
                if not isinstance(z,int) and not isinstance(z,np.integer):
                    if z != 0.0 and z != 1.0:
                         return False
                d=self.counts[v]
                if z in d:
                    d[z]=d[z]+1
                else: d[z]=1
            return True
        else:
            return False


class MultiSetClassification(Classification):

    def __init__(self):
        self.items=set()
        self.counts=dict()
        self.isMulti=False
        pass

    def check(self,item):
        if isinstance(item,list) or isinstance(item,np.ndarray) or isinstance(item,np.ndarray):
            _len=len(item)
            if len(np.unique(item))!=_len:
                return False
            if _len>1:
                self.isMulti=True
            for z in item:
                if not isinstance(z, int) and not isinstance(z, np.integer) and not isinstance(z, np.bool) and not isinstance(z,bool):
                    return False
                if z in self.counts:
                    self.counts[z]=self.counts[z]+1
                else:
                    self.counts[z]=1
            return True
        else:
            return False




class Shapes:

    def __init__(self):
        self.all=[]
        self.shapeSet=set()
        self._count=True

    def add(self,x):
        sh=get_shape(x)
        self.all.append(sh)
        self.shapeSet.add(sh)

    def count(self):
        if self._count is True:
            curCount=None
            for x in self.shapeSet:
                l=len(x)
                co=0
                for c in x:
                    if isinstance(c,tuple):
                        co=co+1
                if co>0:
                    if co!=l:
                        self._count=None
                        return None
                if curCount is None:
                    curCount=l
                elif curCount!=l:
                    self._count = None
                    return None

            if curCount==0:
                self._count=1
                return 1

            self._count=curCount
        return self._count


class ClassBalanceInfo:

    def __init__(self):
        pass

class DatasetTrial:

    def __init__(self,d:datasets.DataSet):
        self.data=d
        self.outputTrials=[BinaryClassification(), MultiClassification(),MultiSetClassification()]
        self.errors = []
        self.perform()

    def addError(self,message,num,extra=None):
        self.errors.append({"message":message,"num":num,"extra":extra})
    def perform(self):
        d_len=len(self.data)
        ids=set()
        inputShapes=Shapes()
        outputShapes = Shapes()
        self.inputShapes=inputShapes
        self.outputShapes=outputShapes
        for i in tqdm(range(0,d_len)):
            try:
                pi:datasets.PredictionItem=self.data[i]
            except:
                exc_type, self.exc_value, self.exc_traceback = sys.exc_info()
                exc_traceback = traceback.format_exc()
                self.addError("Can not access",i,{ "type":str(exc_type),"value":str(self.exc_value),"trace":self.exc_traceback})
                return
            if not isinstance(pi,datasets.PredictionItem):
                self.addError("All data set items should be instances of prediction item",i)
                return
            if pi.id in ids:
                self.addError("All item ids should be unique",i)
                return
            ids.add(pi.id)
            try:
                inputShapes.add(pi.x)
                outputShapes.add(pi.y)
            except ValueError:
                self.addError("Can not determine shape", i)
                return
            toRemove=[]
            for t in self.outputTrials:
                if not t.check(pi.y):
                    toRemove.append(t)
            if len(toRemove)>0:
                self.outputTrials=[x for x in self.outputTrials if x not in toRemove]

            if self.get_inputs_count()==None:
                 self.addError("Can not determine number of inputs", i)
            if self.get_outputs_count()==None:
                 self.addError("Can not determine number of inputs", i)

    def report(self):
        pass

    def get_inputs_count(self):
        return self.inputShapes.count()


    def get_outputs_count(self):
        return self.outputShapes.count()



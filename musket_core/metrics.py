import  keras
import numpy as np
import tqdm
from musket_core import configloader
def keras_metric(func):
    keras.utils.get_custom_objects()[func.__name__]=func
    return func


def final_metric(func):
    func.final_metric=True
    configloader.load("layers").catalog[func.__name__]=func
    return func

SMOOTH = 1e-6


class ByOneMetric:
    
    def __init__(self):
        pass
  
    def onItem(self,outputs,labels):
        pass
    
    def commit(self,dict):
        pass
    
    def eval(self,predictions):
        res={}
        for v in tqdm.tqdm(predictions):
            self.onItem(v.prediction,v.y);
        self.commit(res)    
        return res
    
    def __call__(self,*args):
        if len(args)>1:
            predictions=args[1]
        else:
            predictions=args[0]   
        return self.eval(predictions)

class FunctionalMetric(ByOneMetric):
    
    def __init__(self, function):
        if not hasattr(function, "__name__"):
            self.name=function.__class__.__name__
        else:
            self.name=function.__name__
        pass


class SimpleMetric(FunctionalMetric):    

    def __init__(self,function):
        super().__init__(function)
        self.values=[]
        self.mfunc=function
        pass
  
    def onItem(self,outputs,labels):
        self.values.append(self.mfunc(outputs,labels))
        pass
    
    def commit(self,dict):
        dict[self.name]=np.mean(self.values)        


class WithTreshold(FunctionalMetric):    

    def __init__(self,function,cnt=100.0):
        super().__init__(function)
        self.vals=[[] for x in range(int(cnt))]
        self.cnt=cnt
        self.mfunc=function
        pass
  
    def onItem(self,outputs,labels):
        for i in range(0,round(self.cnt)):
            val=self.mfunc(outputs>i*(1/self.cnt),labels)
            self.vals[i].append(val)
    
    def commit(self,dict):
        ms=[np.mean(v) for v in self.vals]
        ms=np.array(ms)
        ma = np.where(ms == np.max(ms))[0]
        if len(ma)>0:
            ma=ma[0]
        tr= float(ma)*(1/self.cnt)
        bestVal=float(np.max(ms))
        dict[self.name+"_treshold"]=tr
        dict[self.name]=bestVal
        
def Composite(ByOneMetric):            
    def __init__(self,items):
        self.items=items
    
    def onItem(self,outputs,labels):
        for x in self.items:
            x.onItem(outputs,labels)
            
    def commit(self,dict):           
        for x in self.items:
            x.commit(dict) 


class Dice:
    def __init__(self,zero_is):
        self.zero_is=zero_is
        
    def __call__(self,outputs,labels):    
            cs=labels.shape[-1]
            if cs>1:
                rr=[]
                for i in range(cs):
                    rr.append(self(outputs[:,:,i:i+1], labels[:,:,i:i+1]))
                return np.mean(rr, axis=0)
            outputs = outputs.squeeze()
            labels = labels.squeeze()
            true = (outputs>0.5)
            pred = (labels>0.5)
            
            if labels.max()==0:
                if (outputs>0.5).max()==0:
                    return self.zero_is
            return self.innerCalc(true,pred)
    
    def innerCalc(self,true,pred):          
        intersection =np.sum (true & pred)
        im_sum = np.sum(true) + np.sum(pred)
        
        return float(2.0 * intersection / (im_sum + SMOOTH))
    
class IOU(Dice):    
    
    def innerCalc(self,labels,outputs):          
        intersection = (outputs & labels).sum()
        union = (outputs | labels).sum()
        
        iou = (intersection + SMOOTH) / (union + SMOOTH)
        return float(iou)
        
class MAP10(Dice):
    def innerCalc(self,labels,outputs):          
        intersection = (outputs & labels).sum()
        union = (outputs | labels).sum()
        
        iou = (intersection + SMOOTH) / (union + SMOOTH)
        
        thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
        
        return float(thresholded)  # Or thresholded.mean()

                

metricNames=["dice","iou","map10"]

ms={"dice":Dice,"iou":IOU,"map10":MAP10}
treshold=[True,False]
zero_is=[0,1,None]
for m in metricNames:
    for t in treshold:
        for z in zero_is:
            mName=m
            if t:
                mName=mName+"_with_custom_treshold_"
            else:
                mName=mName+"_"
            
            if z==0:
                mName=mName+"true_negative_is_zero"    
            elif z==1:
                mName=mName+"true_negative_is_one"
            else:
                mName=mName+"true_negative_is_skip"
            fnc=ms[m](z)
            fnc.__name__=mName
            if t:
                fnc=WithTreshold(fnc)    
            else:
                fnc=SimpleMetric(fnc)    
            configloader.load("layers").catalog[fnc.name]=fnc                                
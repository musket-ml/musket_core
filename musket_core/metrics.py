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

from musket_core.losses import dice_numpy,iou_coef_numpy,dice_numpy_zero_is_one


def map10_binary_numpy(outputs: np.array, labels: np.array,emptyIsOne=False):
    if emptyIsOne:
        if labels.max()==0:
            if (outputs>0.5).max()>0:
                return 0
            return 1 
    outputs = outputs.squeeze()>0.3
    labels = labels.squeeze()>0.5
    
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()    

def map10_binary_empty_is_1_numpy(outputs: np.array, labels: np.array):
    return map10_binary_numpy(outputs,labels,True)



def byOne(predictions,func):
    vals=[]
    for v in tqdm.tqdm(predictions):
        val=func(v.prediction,v.y)
        vals.append(val)
    return np.mean(vals)

def findTreshold(predictions,func):
    vals=[[] for x in range(100)]
    for v in tqdm.tqdm(predictions,"Estimating optimal treshold"):
        for i in range(0,100):
            val=func(v.prediction>i*0.01,v.y)
            vals[i].append(val)
    ms=[np.mean(v) for v in vals]
    ms=np.array(ms)
    return float(np.where(ms==np.max(ms))[0])*0.01,np.max(ms)

@final_metric
def map10(*args):
    predictions=args[1]
    return byOne(predictions, map10_binary_empty_is_1_numpy)

@final_metric
def dice1(*args):
    predictions=args[1]
    return byOne(predictions, dice_numpy)

@final_metric
def dice_true_negative_is_1(*args):
    if len(args)>1:
        predictions=args[1]
    else:
        predictions=args[0]    
    return byOne(predictions, dice_numpy_zero_is_one)

@final_metric
def iou(*args):
    predictions=args[1]
    return byOne(predictions, iou_coef_numpy)

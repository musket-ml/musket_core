from musket_core import generic_config
from musket_core.utils import ensure
import os
import numpy as np
from musket_core.structure_constants import constructPredictionsDirPath
from musket_core.datasets import DataSet, PredictionItem,PredictionBlend
from musket_core.preprocessing import PreprocessedDataSet
from typing import Union
import keras
from musket_core import configloader,metrics
from musket_core.model import IGenericTaskConfig,FoldsAndStages
from tqdm import tqdm
import tensorflow as tf

import keras.backend as K
class Prediction:
    def __init__(self,cfg,fold,stage,name:str,srcDataset=None):
        fold, stage = _fix_fold_and_stage(cfg, fold, stage)
        self.cfg = cfg
        self.fold = fold
        self.stage=stage
        self.name=name        
        self.srcDataset=srcDataset

    def calculate(self)->DataSet:    
        ds = self.srcDataset
        nm = self.name
        if ds is None:
            if self.name is not None and isinstance(self.name, str):
                if self.name=="holdout":
                    ds=self.cfg.holdout()
                elif self.name=="validation":
                    ds=self.cfg.validation(None,self.fold)
                else:
                    ds = self.cfg.get_dataset(self.name)
        else:
            nm = ds.name

        if ds is None:
            raise ValueError("No dataset has been specified for prediction")

        ensure(constructPredictionsDirPath(self.cfg.directory()))
        path = f"{constructPredictionsDirPath(self.cfg.directory())}/{nm}{str(self.stage)}{str(self.fold)}.npy"

        if os.path.exists(path):
            return self.cfg.load_writeable_dataset(ds, path)
        
        if self.cfg.separatePredictions:
            if isinstance(self.fold, list) and len(self.fold)>1:
                prs=[Prediction(self.cfg,v,self.stage,self.name,self.srcDataset).calculate() for v in self.fold]
                vls=PredictionBlend(prs)    
                wd=self.cfg.create_writeable_dataset(vls,path)
                for i in tqdm(vls,"Blending"):
                    wd.append(i.prediction)
                wd.commit()    
                return self.cfg.load_writeable_dataset(ds, path)
            if isinstance(self.stage, list) and len(self.stage)>1:
                prs=[Prediction(self.cfg,self.fold,v,self.name,self.srcDataset).calculate() for v in self.stage]
                vls=PredictionBlend(prs)
                wd=self.cfg.create_writeable_dataset(vls,path)
                for i in tqdm(vls,"Blending"):
                    wd.append(i.prediction)
                wd.commit()    
                return self.cfg.load_writeable_dataset(ds, path)                
            
            pass
        if self.cfg.needsSessionForPrediction:            
            #K.clear_session()
            try:
                with self.create_session().as_default():
                    value=self.cfg.predict_all_to_dataset(ds,self.fold,self.stage,-1,None,False,path)
            finally:        
                K.clear_session()
        else:    
            value=self.cfg.predict_all_to_dataset(ds,self.fold,self.stage,-1,None,False,path)
        return value
    def create_session(self):
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        )
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess

def get_validation_prediction(cfg,fold:int,stage=None)->DataSet:
    if stage is None:
        stage=list(range(len(cfg.stages)))
    return Prediction(cfg,fold,stage,"validation").calculate()


def get_holdout_prediction(cfg,fold=None,stage=None)->DataSet:
    return Prediction(cfg,fold,stage,"holdout").calculate()


def _fix_fold_and_stage(cfg, fold, stage):
    if stage is None:
        stage = list(range(len(cfg.stages)))
    if fold is None:
        fold = list(range(cfg.folds_count))
    if isinstance(stage, int): 
        stage=[stage]
    if isinstance(fold, list):
        if len(fold)==1: 
            fold=fold[0]
    return fold, stage


def get_predictions(cfg,name_or_ds:Union[str,DataSet],fold=None,stage=None)->DataSet:
    if isinstance(name_or_ds   , str):
        return Prediction(cfg,fold,stage,name_or_ds,None).calculate()
    else: return Prediction(cfg,fold,stage,None,name_or_ds).calculate()

def stat(metrics):
    return {"mean":float(np.mean(metrics)),"max":float(np.max(metrics)),"min":float(np.min(metrics)),"std":float(np.std(metrics))}

def isFinal(metric:str)->bool:
    try:
        mtr = keras.metrics.get(metric)
        return False
    except:
        return True

def cross_validation_stat(cfg:IGenericTaskConfig, metric, stage=None, threshold=0.5, folds=None):
    metrics=[]
    cfg.get_dataset()# this is actually needed
    mDict={}
    if not folds:
        folds = list(range(cfg.folds_count))
        if hasattr(cfg, "_folds"):
            folds=cfg._folds()
    if isFinal(metric):
        fnc=configloader.load("layers").catalog[metric]
        if hasattr(fnc, "func"):
            fnc=fnc.func    
        for i in folds:
            fa=FoldsAndStages(cfg,i,stage)
            val=cfg.predictions("validation",i,stage)
            pv = fnc(fa, val)
            if isinstance(pv, dict):
                for k in pv:
                    if k not in mDict:
                        mDict[k]=[]
                    mDict[k].append(pv[k])    
            else:
                metrics.append(pv)
        if len(mDict)>0:
            rs={}
            for c in mDict:
                rs[c]=stat(mDict[c])
            return rs            
        return stat(metrics)

    for i in folds:
        if cfg._reporter is not None and cfg._reporter.isCanceled():
            return {"canceled": True}

        predictionDS = get_validation_prediction(cfg, i, stage)
        val = considerThreshold(predictionDS, metric, threshold)
        eval_metric = generic_config.eval_metric(val, metric, cfg.get_eval_batch())
        metrics.append(np.mean(eval_metric))
    return stat(metrics)


def holdout_stat(cfg:IGenericTaskConfig, metric,stage=None,threshold=0.5):
    if cfg._reporter is not None and cfg._reporter.isCanceled():
        return {"canceled": True}
    
    if isFinal(metric):
        fnc=configloader.load("layers").catalog[metric]
        if hasattr(fnc, "func"):
            fnc=fnc.func    
        for i in range(cfg.folds_count):
            fa=FoldsAndStages(cfg,i,stage)
            val=cfg.predictions("holdout",i,stage)
            r = fnc(fa, val)
            if isinstance(r, dict):
                return r
            return float(r)
    predictionDS = get_holdout_prediction(cfg, None, stage)
    val = considerThreshold(predictionDS, metric, threshold)
    eval_metric = generic_config.eval_metric(val, metric, cfg.get_eval_batch())
    if cfg._reporter is not None and cfg._reporter.isCanceled():
        return {"canceled": True}
    return float(np.mean(eval_metric))


def considerThreshold(predictionDS, metric, threshold)->DataSet:

    need_threshold = generic_config.need_threshold(metric)
    if need_threshold:
        def applyThreshold(dsItem: PredictionItem) -> PredictionItem:
            thresholded = dsItem.prediction > threshold
            result = PredictionItem(dsItem.id, dsItem.x, dsItem.y, thresholded)
            return result

        val = PreprocessedDataSet(predictionDS, applyThreshold, True)
    else:
        val = predictionDS
    return val
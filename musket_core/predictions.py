from musket_core import generic_config
from musket_core.utils import ensure
import os
import numpy as np
from musket_core.structure_constants import constructPredictionsDirPath
from musket_core.datasets import DataSet, PredictionItem
from musket_core.preprocessing import PreprocessedDataSet
from typing import Union

from musket_core.model import IGenericTaskConfig

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

        value=self.cfg.predict_all_to_dataset(ds,self.fold,self.stage,-1,None,False,path)
        return value


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
    return fold, stage


def get_predictions(cfg,name_or_ds:Union[str,DataSet],fold=None,stage=None)->DataSet:
    if isinstance(name_or_ds   , str):
        return Prediction(cfg,fold,stage,name_or_ds,None).calculate()
    else: return Prediction(cfg,fold,stage,None,name_or_ds).calculate()

def stat(metrics):
    return {"mean":float(np.mean(metrics)),"max":float(np.max(metrics)),"min":float(np.min(metrics)),"std":float(np.std(metrics))}


def cross_validation_stat(cfg:IGenericTaskConfig, metric,stage=None,treshold=0.5):
    metrics=[]
    cfg.get_dataset()
    for i in range(cfg.folds_count):
        if cfg._reporter is not None and cfg._reporter.isCanceled():
            return {"canceled": True}
        predictionDS = get_validation_prediction(cfg, i, stage)
        val = considerThreshold(predictionDS, metric, treshold)
        eval_metric = generic_config.eval_metric(val, metric, cfg.get_eval_batch())
        metrics.append(np.mean(eval_metric))
    return stat(metrics)


def holdout_stat(cfg:IGenericTaskConfig, metric,stage=None,treshold=0.5):
    if cfg._reporter is not None and cfg._reporter.isCanceled():
        return {"canceled": True}
    predictionDS = get_holdout_prediction(cfg, None, stage)
    val = considerThreshold(predictionDS, metric, treshold)
    eval_metric = generic_config.eval_metric(val, metric, cfg.get_eval_batch())
    if cfg._reporter is not None and cfg._reporter.isCanceled():
        return {"canceled": True}
    return float(np.mean(eval_metric))


def considerThreshold(predictionDS, metric, treshold)->DataSet:

    need_threshold = generic_config.need_threshold(metric)
    if need_threshold:
        def applyThreshold(dsItem: PredictionItem) -> PredictionItem:
            thresholded = dsItem.prediction > treshold
            result = PredictionItem(dsItem.id, dsItem.x, dsItem.y, thresholded)
            return result

        val = PreprocessedDataSet(predictionDS, applyThreshold, True)
    else:
        val = predictionDS
    return val
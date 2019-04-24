from musket_core import generic_config
from musket_core.utils import ensure
import os
import numpy as np
from musket_core.structure_constants import constructPredictionsDirPath
from musket_core.datasets import DataSet, PredictionItem, WrappedDS
from musket_core.preprocessing import PreprocessedDataSet

class Prediction:
    def __init__(self,cfg,fold,stage,name:str):
        fold, stage = _fix_fold_and_stage(cfg, fold, stage)
        self.cfg = cfg
        self.fold = fold
        self.stage=stage
        self.name=name

    def calculate(self)->DataSet:
        ensure(constructPredictionsDirPath(self.cfg.directory()))
        nm=self.name
        if not isinstance(self.name,str):
            if hasattr(self.name,"name"):
                nm=getattr(self.name,"name")
            if hasattr(self.name,"origName"):
                nm=getattr(self.name,"origName")
        path = f"{constructPredictionsDirPath(self.cfg.directory())}/{nm}{str(self.stage)}{str(self.fold)}.npy"

        if not isinstance(self.name,str):
            ds=self.name
        elif self.name=="holdout":
            ds=self.cfg.holdout()
        elif self.name=="validation":
            ds=self.cfg.validation(None,self.fold)
        else:
            ds=self.cfg.get_dataset(self.name)
        if os.path.exists(path):
            rr= np.load(path)
            resName = (ds.name if hasattr(ds, "name") else "") + "_predictions"
            result = WrappedDS(ds, [i for i in range(len(ds))], resName, None, rr)
            return result
        value=self.cfg.predict_all_to_dataset(ds,self.fold,self.stage)
        np.save(path,value.predictions)
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


def get_predictions(cfg,name,fold=None,stage=None)->DataSet:
    return Prediction(cfg,fold,stage,name).calculate()

def stat(metrics):
    return {"mean":float(np.mean(metrics)),"max":float(np.max(metrics)),"min":float(np.min(metrics)),"std":float(np.std(metrics))}


def cross_validation_stat(cfg, metric,stage=None,treshold=0.5):
    metrics=[]
    cfg.get_dataset()
    for i in range(cfg.folds_count):
        if cfg._reporter is not None and cfg._reporter.isCanceled():
            return {"canceled": True}
        predictionDS = get_validation_prediction(cfg, i, stage)
        val = considerThreshold(predictionDS, metric, treshold)
        eval_metric = generic_config.eval_metric(val, metric, cfg.inference_batch)
        metrics.append(np.mean(eval_metric))
    return stat(metrics)


def holdout_stat(cfg, metric,stage=None,treshold=0.5):
    if cfg._reporter is not None and cfg._reporter.isCanceled():
        return {"canceled": True}
    predictionDS = get_holdout_prediction(cfg, None, stage)
    val = considerThreshold(predictionDS, metric, treshold)
    eval_metric = generic_config.eval_metric(val, metric, cfg.inference_batch)
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
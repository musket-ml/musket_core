from musket_core import generic_config
from musket_core.utils import ensure
import os
import numpy as np
from musket_core import datasets


class Prediction:
    def __init__(self,cfg,fold,stage,name:str):
        fold, stage = _fix_fold_and_stage(cfg, fold, stage)
        self.cfg = cfg
        self.fold = fold
        self.stage=stage
        self.name=name

    def calculate(self):
        ensure(self.cfg.directory()+"/predictions")
        path = self.cfg.directory() + "/predictions/" + self.name + str(self.stage) + str(self.fold) + ".npy"
        if os.path.exists(path):
            rr= np.load(path)
            return rr
        ds=None
        if self.name=="holdout":
            ds=self.cfg.holdout()
        elif self.name=="validation":
            ds=self.cfg.validation(self.fold)
        else:
            ds=self.cfg.get_dataset(self.name)
        value=self.cfg.predict_all_to_array(ds,self.fold,self.stage)
        np.save(path,value)
        return value


def get_validation_prediction(cfg,fold:int,stage=None):
    if stage is None:
        stage=list(range(len(cfg.stages)))
    return Prediction(cfg,fold,stage,"validation").calculate()


def get_holdout_prediction(cfg,fold=None,stage=None):
    return Prediction(cfg,fold,stage,"holdout").calculate()


def _fix_fold_and_stage(cfg, fold, stage):
    if stage is None:
        stage = list(range(len(cfg.stages)))
    if fold is None:
        fold = list(range(cfg.folds_count))
    return fold, stage


def get_predictions(cfg,name,fold=None,stage=None):
    return Prediction(cfg,fold,stage,name).calculate()

def stat(metrics):
    return {"mean":float(np.mean(metrics)),"max":float(np.max(metrics)),"min":float(np.min(metrics)),"std":float(np.std(metrics))}


def cross_validation_stat(cfg, metric,stage=None,treshold=0.5):
    metrics=[]
    for i in range(cfg.folds_count):
        val=get_validation_prediction(cfg,i,stage)>treshold
        vt=datasets.get_targets_as_array(cfg.validation(i))
        eval_metric = generic_config.eval_metric(vt, val, metric)
        metrics.append(np.mean(eval_metric))
    return stat(metrics)

def holdout_stat(cfg, metric,stage=None,treshold=0.5):
    val=get_holdout_prediction(cfg,None,stage)>treshold
    vt=datasets.get_targets_as_array(cfg.holdout())
    eval_metric = generic_config.eval_metric(vt, val, metric)
    return float(np.mean(eval_metric))
from musket_core import datasets,generic_config
import numpy as np


def block(func):
    func.model=True
    return func

class Model:

    def predict_on_dataset(self,d:datasets.DataSet,**kwargs):
        raise ValueError("Not implemented")


class ConnectedModel(Model):

    def predictions(self,name,**kwargs):
        raise ValueError("Not implemented")




class FoldsAndStages(ConnectedModel):

    def __init__(self,core,folds,stages):
        self.wrapped=core
        self.folds=folds
        self.stages=stages

    def predict_on_dataset(self, d, **kwargs):
        return self.wrapped.predict_on_dataset(d,fold=self.folds,stage=self.stages)

    def predictions(self,name, **kwargs):
        return self.wrapped.predictions(name, fold=self.folds, stage=self.stages)

class AverageBlend(ConnectedModel):

    def __init__(self,models):
        self.models=models
        pass
    
    
    def predict_on_dataset(self, d, **kwargs):
        raise ValueError("Not Implemented")

    def predictions(self,name, **kwargs):
        rm=[]
        for v in self.models:
            rm.append(v.predictions(name))
        return np.array(rm).mean(axis=0)    
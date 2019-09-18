from musket_core.datasets import DataSet,PredictionItem,dataset_provider
import pandas as pd
import numpy as np
import os
from musket_core.context import get_current_project_path,\
    get_current_project_data_path

class CSVDataSet(DataSet):
    
    def __init__(self,path,targetColumn:str,featureColumn,idColumn:str=None,sep=","): 
        self.df=pd.read_csv(path,sep=sep)
        self.feature=self.df[featureColumn].values
        
        if (targetColumn in self.df.columns):
            self._target=self.df[targetColumn].values
        else:
            self._target=np.zeros((len(self),1))
        if idColumn is not None:    
            self.ids=self.df[idColumn].values        
        else:
            self.ids=list(range(len(self.df)))   
        pass    
    
    def __len__(self):
        return round(len(self.df))
        
    def __getitem__(self, item)->PredictionItem:
        return PredictionItem(self.ids[item],self.feature[item],np.array([self._target[item]]))
    
    def get_target(self,item):
        return np.array([self._target[item]])
    
@dataset_provider    
def from_csv(path,targetColumn:str,featureColumn:str,idColumn:str=None,sep=",",absPath=False):
    if not absPath:
        path=os.path.join(get_current_project_data_path(),path) 
    return CSVDataSet(path,targetColumn,featureColumn,idColumn,sep)


class FromArrays(DataSet):
    
    def __init__(self,x,y):
        self.x=x
        self.y=y    
        
    def __len__(self):
        return round(len(self.x))
        
    def __getitem__(self, item)->PredictionItem:
        return PredictionItem(item,self.x[item],self.y[item])
    
    def get_target(self,item):
        return self.y[item]
    
def from_array(x,y):
    return FromArrays(x,y)        
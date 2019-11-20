from musket_core import generic,context,genericcsv,configloader,parralel
import pandas as pd
import os
import sys
import importlib
import numpy as np

def _from_numpy(x):
    if isinstance(x, np.ndarray):
        return [_from_numpy(i) for i in x]
    if isinstance(x, np.bool):
        return bool(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.float):
        return float(x)
    if isinstance(x, np.int):
        return float(x)
    if isinstance(x, list):
        return [_from_numpy(i) for i in x]
    if isinstance(x, dict):
        res={}
        for (k,v) in x.items():
            ds=_from_numpy(v)
            if "|" in k:
                clns=k.split("|")
                for c in clns:
                    if c in ds:
                        res[c]=True
                    else:
                        res[c]=False    
                pass
            else:
                res[k]=ds 
        return res            
    return x
            

class BasicEngine:
    
    def __init__(self,path,input_columns,output_columns,ctypes={},input_groups={},output_groups={}):
        self.cfg=generic.parse(path)
        path=os.path.dirname(os.path.dirname(path))
        self.cfg._projectDir=path
        self.input_columns=input_columns
        self.output_columns=output_columns
        self.image_path=[]
        self.ctypes=ctypes
        self.path=path
        self.input_groups=input_groups
        self.output_groups=output_groups
        
        
    def __call__(self,inp):
        unpack=False
        if isinstance(inp, dict):
            inp= [inp]
            unpack=True
        pi=pd.DataFrame(inp)
        context.context.no_cache=True   
        context.context.projectPath=self.path
        context.context.dataPath=os.path.join(self.path,"assets")
        dataset=genericcsv.GenericCSVDataSet(pi,self.input_columns,self.output_columns,[],self.ctypes,self.input_groups,self.output_groups)
        
        dataset.ignoreOutput=True
        dataset.init_coders_from_path(self.path)
        res=self.cfg.predict_all_to_dataset(dataset, cacheModel=True,verbose=0)
        result_frame=dataset.encode(res)        
        for c in self.output_columns:
            vls=result_frame[c].values
            for i in range(len(inp)):
                inp[i][c]=vls[i]
        inp=_from_numpy(inp)        
        if unpack:
            return inp[0]        
        return inp
    
def inference_service_factory(f):
    f.serviceCreator=True
    return f
    
def create_engine(path:str,multi_threaded=False):    
    sys.path.insert(0, path)
    creator=None
    for f in os.listdir(path):
        if len(f)>3 and f[-3:]==".py":
            mod=importlib.import_module(f[:-3])
            configloader.register(mod)             
            for x in dir(mod):
                ele=getattr(mod, x)
                if hasattr(ele,"serviceCreator"):
                    creator=ele
    try:
        from musket_text import preprocessors
        configloader.register(preprocessors)
    except:
        pass                    
    if multi_threaded:
        engine=creator()                
        def fnc(data):
            def calc():
                return engine(data)
            t=parralel.Task(calc,requiresSession=False)
            parralel.schedule([t])
            return t.result
        return fnc
    return creator()                


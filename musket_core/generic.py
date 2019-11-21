import musket_core.generic_config as generic
import musket_core.datasets as datasets
import musket_core.configloader as configloader
import musket_core.utils as utils
import musket_core.context as context
import numpy as np
import keras
import musket_core.net_declaration as net
import musket_core.quasymodels as qm
import os
import tqdm
import sys



def model_function(func):
    func.model=True
    return func

def _shape(x):
    if isinstance(x,tuple):
        return [i.shape for i in x]
    if isinstance(x,list):
        return [i.shape for i in x]
    return x.shape

class GenericPipeline(generic.GenericTaskConfig):

    def __init__(self,**atrs):
        super().__init__(**atrs)
        self.dataset_clazz = datasets.DefaultKFoldedDataSet
        self._multiOutput=None
        pass

    def createNet(self):
        inp,output=utils.load_yaml(self.path + ".shapes")
        if not hasattr(context.context,"net_cx"):
            context.context.net_cx=[]
        contributions=None
        if os.path.exists(self.path+".contribution"):
            contributions=utils.load(self.path+".contribution")
        else:
            contributions=None    
        if isinstance(inp,list):
            
            inputs=[keras.Input(x) for x in inp]
            if contributions is not None:
                if isinstance(contributions, list):
                    for i in range(len(inputs)):
                        inputs[i].contribution=contributions[i]
                else:                   
                    for i in range(len(inputs)):
                        inputs[i].contribution=contributions
        else:
            i=keras.Input(inp);
            i.contribution=contributions
            inputs=[i]
        m=net.create_model_from_config(self.declarations,inputs,self.architecture,self.imports)
        if context.isTrainMode():
            if hasattr(context.context, "net_cx"):
                utils.save(self.path+".ncx", context.context.net_cx)
        context.context.net_cx=[]
        
        return m

    def load_writeable_dataset(self, ds, path):
        if self.isMultiOutput():
            rr = utils.load(path)
            resName = (ds.name if hasattr(ds, "name") else "") + "_predictions"
            result = datasets.BufferedWriteableDS(ds, resName, path, rr)
        else:
            rr = np.load(path)
            resName = (ds.name if hasattr(ds, "name") else "") + "_predictions"
            result = datasets.BufferedWriteableDS(ds, resName, path, rr)
        return result



    def create_writeable_dataset(self, dataset:datasets.DataSet, dsPath:str)->datasets.WriteableDataSet:
        inp,output=utils.load_yaml(self.path + ".shapes")
        resName = (dataset.name if hasattr(dataset, "name") else "") + "_predictions"
        result = datasets.BufferedWriteableDS(dataset, resName, dsPath,pickle=self.isMultiOutput())
        return result

    def isMultiOutput(self):
        if self._multiOutput is not None:
            return self._multiOutput
        inp,output=utils.load_yaml(self.path + ".shapes")
        self._multiOutput= len(output)>1 and isinstance(output, list)
        return self._multiOutput

    def predict_on_batch(self, mdl, ttflips, batch):

        res =  mdl.predict(batch.images)
        if self.isMultiOutput():
            result=[]
            for i in range(len(res[0])):
                elementOutputs=[]
                for x in res:
                    elementOutputs.append(x[i])
                result.append(elementOutputs)
            return result

        return res

    def evaluateAll(self,ds, fold:int,stage=-1,negatives="real",ttflips=None,batchSize=32):
        folds = self.kfold(ds, range(0, len(ds)),batch=batchSize)
        vl, vg, test_g = folds.generator(fold, False,negatives=negatives,returnBatch=True)
        indexes = folds.sampledIndexes(fold, False, negatives)
        m = self.load_model(fold, stage)
        num=0
        with tqdm.tqdm(total=len(indexes), unit="files", desc="segmentation of validation set from " + str(fold)) as pbar:
            try:
                for f in test_g():
                    if num>=len(indexes): break
                    x, y, b = f
                    z = self.predict_on_batch(m,ttflips,b)
                    ids=b.data[0]
                    b.results=z
                    b.ground_truth=b.data[1]
                    yield b
                    num=num+len(z)
                    pbar.update(len(ids))
            finally:
                vl.terminate()
                vg.terminate()
        pass

    def evaluate_all_to_arrays(self,ds, fold:int,stage=-1,negatives="real",ttflips=None,batchSize=32):
        lastFullValPred = None
        lastFullValLabels = None
        for v in self.evaluateAll(ds, fold, stage,negatives,ttflips,batchSize):
            if lastFullValPred is None:
                lastFullValPred = v.results
                lastFullValLabels = v.ground_truth
            else:
                lastFullValPred = np.append(lastFullValPred, v.results, axis=0)
                lastFullValLabels = np.append(lastFullValLabels, v.ground_truth, axis=0)
        return lastFullValPred,lastFullValLabels

    def predict_on_dataset(self, dataset, fold=0, stage=0, limit=-1, batch_size=32, ttflips=False, cacheModel=False):
        if cacheModel:
            if hasattr(self, "_mdl"):
                mdl=self._mdl
            else:    
                mdl = self.createNetForInference(fold, stage)
                self._mdl=mdl
        else:
            mdl = self.createNetForInference(fold, stage)       
        if self.testTimeAugmentation is not None:
            mdl=qm.TestTimeAugModel(mdl,net.create_test_time_aug(self.testTimeAugmentation,self.imports))
        if self.preprocessing is not None:
            dataset = net.create_preprocessor_from_config(self.declarations, dataset, self.preprocessing, self.imports)
        for original_batch in datasets.generic_batch_generator(dataset, batch_size, limit):
            res = self.predict_on_batch(mdl, ttflips, original_batch)
            original_batch.results=res
            yield original_batch

    def predict_in_dataset(self, dataset, fold, stage, cb, data, limit=-1, batch_size=32, ttflips=False):
        with tqdm.tqdm(total=len(dataset), unit="files", desc="prediction from  " + str(dataset)) as pbar:
            for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips):
                b=v
                for i in range(len(b.data)):
                    id=b.data[i]
                    cb(id,b.results[i],data)
                pbar.update(batch_size)



    def predict_all_to_array_with_ids(self, dataset, fold, stage, limit=-1, batch_size=32, ttflips=False):
        res=[]
        ids=[]
        with tqdm.tqdm(total=len(dataset), unit="files", desc="prediction from  " + str(dataset)) as pbar:
            for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips):
                b=v
                for i in range(len(b.data)):
                    id=b.data[i]
                    ids.append(id)
                    res.append(b.results[i])
                pbar.update(batch_size)
        return np.array(res),ids

    def fit(self, dataset=None, subsample=1.0, foldsToExecute=None, start_from_stage=0, drawingFunction=None,parallel = False):
        dataset = self.init_shapes(dataset)
        return super().fit(dataset,subsample,foldsToExecute,start_from_stage,drawingFunction,parallel=parallel)

    def validate(self):
        self.init_shapes(None)
        super().validate()

    def init_shapes(self, dataset):
        if dataset is None:
            dataset = self.get_dataset()
        self._dataset = dataset
        if self.preprocessing is not None:
            dataset = net.create_preprocessor_from_config(self.declarations, dataset, self.preprocessing, self.imports)
        predItem = dataset[0]
        if hasattr(dataset, "contribution"):
            utils.save(self.path+ ".contribution",getattr(dataset, "contribution"))
        elif hasattr(dataset, "contributions"):
            utils.save(self.path+ ".contribution",getattr(dataset, "contributions"))
        utils.save_yaml(self.path + ".shapes", (_shape(predItem.x), _shape(predItem.y)))
        return dataset


def parse(path,extra=None) -> GenericPipeline:
    extraImports=[]
    if isinstance(path, str):
        if not os.path.exists(path) or os.path.isdir(path):
            pth=context.get_current_project_path()
            if os.path.exists(pth+"/experiments/"+path+"/config.yaml"):
                path=pth+"/experiments/"+path+"/config.yaml"
            if os.path.exists(pth+"/common.yaml") and extra is None:
                extra=pth+"/common.yaml"
            if os.path.exists(pth+"/modules"):
                for m in os.listdir(pth+"/modules"):
                    sys.path.insert(0, pth+"/modules")
                    if ".py" in m:
                        extraImports.append(m[0:m.index(".py")])   
    cfg = configloader.parse("generic", path,extra)
    cfg.path = path    
    for e in extraImports:
        cfg.imports.append(e)
    return cfg
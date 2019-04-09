import musket_core.generic_config as generic
import musket_core.datasets as datasets
import musket_core.configloader as configloader
import musket_core.utils as utils
import numpy as np
import keras
import tqdm
import musket_core.net_declaration as net
import musket_core.quasymodels as qm
import os

def model_function(func):
    func.model=True
    return func

def _shape(x):
    if isinstance(x,tuple):
        return [i.shape for i in x]
    if isinstance(x,list):
        return np.array(x).shape
    return x.shape

class GenericPipeline(generic.GenericTaskConfig):

    def __init__(self,**atrs):
        super().__init__(**atrs)
        self.dataset_clazz = datasets.DefaultKFoldedDataSet

        pass

    def createNet(self):
        input,output=utils.load_yaml(self.path + ".shapes")
        if isinstance(input,list):
            inputs=[keras.Input(x) for x in input]
        else:
            inputs=[keras.Input(input)]
        m=net.create_model_from_config(self.declarations,inputs,self.architecture,self.imports)
        return m

    def predict_on_batch(self, mdl, ttflips, batch):

        res = mdl.predict(np.array(batch.images))
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

    def predict_on_dataset(self, dataset, fold=0, stage=0, limit=-1, batch_size=32, ttflips=False):
        mdl = self.load_model(fold, stage)
        if self.testTimeAugmentation is not None:
            mdl=qm.TestTimeAugModel(mdl,net.create_test_time_aug(self.testTimeAugmentation,self.imports))
        if self.preprocessing is not None:
            dataset = net.create_preprocessor_from_config(self.declarations, dataset, self.preprocessing, self.imports)
        for original_batch in datasets.batch_generator(dataset, batch_size, limit):
            res = self.predict_on_batch(mdl, ttflips, original_batch)
            original_batch.results=res
            yield original_batch

    def predict_in_dataset(self, dataset, fold, stage, cb, data, limit=-1, batch_size=32, ttflips=False):
        with tqdm.tqdm(total=len(dataset), unit="files", desc="prediiction from  " + str(dataset)) as pbar:
            for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips):
                b=v
                for i in range(len(b.data)):
                    id=b.data[i]
                    cb(id,b.results[i],data)
                pbar.update(batch_size)



    def predict_all_to_array_with_ids(self, dataset, fold, stage, limit=-1, batch_size=32, ttflips=False):
        res=[]
        ids=[]
        with tqdm.tqdm(total=len(dataset), unit="files", desc="prediiction from  " + str(dataset)) as pbar:
            for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips):
                b=v
                for i in range(len(b.data)):
                    id=b.data[i]
                    ids.append(id)
                    res.append(b.results[i])
                pbar.update(batch_size)
        return np.array(res),ids

    def parse_dataset(self,name=None):
        fw=self.dataset
        if self.datasets_path is not None:
            os.chdir(self.datasets_path) #TODO review
        if name is not None:
            fw=self.datasets[name]
        if isinstance(fw,str):
            fw = self.datasets[name]
        if self.dataset is not None:
            dataset = net.create_dataset_from_config(self.declarations, fw,  self.imports)
            if self.preprocessing is not None and self.preprocessing!="":
                dataset = net.create_preprocessor_from_config(self.declarations, dataset, self.preprocessing,
                                                              self.imports)
            return dataset
        return None


    def fit(self, dataset=None, subsample=1.0, foldsToExecute=None, start_from_stage=0, drawingFunction=None,parallel = False):
        if dataset is None:
          dataset = self.get_dataset()
        self._dataset = dataset
        if self.preprocessing is not None:
            dataset = net.create_preprocessor_from_config(self.declarations, dataset, self.preprocessing, self.imports)
        predItem=dataset[0]
        utils.save_yaml(self.path+".shapes",(_shape(predItem.x),_shape(predItem.y)))
        return super().fit(dataset,subsample,foldsToExecute,start_from_stage,drawingFunction,parallel=parallel)


def parse(path,extra=None) -> GenericPipeline:
    cfg = configloader.parse("generic", path,extra)
    cfg.path = path    
    return cfg
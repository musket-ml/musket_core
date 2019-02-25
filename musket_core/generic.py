import musket_core.generic_config as generic
import musket_core.datasets as datasets
import musket_core.configloader as configloader
import numpy as np
import keras
import tqdm


def createSequence(seq):
    layers=configloader.load("layers")
    layerMap={}
    pName=[]
    first=True

    input=None
    for v in seq:

        layer=layers.instantiate(v,True)[0]
        name= v["name"] if "name" in v else layer.name

        inputs=v["inputs"] if "inputs" in v else pName

        prev=layerMap[pName] if isinstance(inputs, str) else [layerMap[l]for l in inputs]

        if first:
            ls = layer
            input=layer
        else: ls = layer(prev)
        layerMap[name]=ls
        first=False
        pName=name

    return keras.Model(input,layerMap[pName])    

class GenericPipeline(generic.GenericTaskConfig):

    def __init__(self,**atrs):
        super().__init__(**atrs)
        self.dataset_clazz = datasets.DefaultKFoldedDataSet
        pass

    def createNet(self):
        return createSequence(self.architecture)

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
                    b.results=z;
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


    def predict_in_dataset(self, dataset, fold, stage, cb, data, limit=-1, batch_size=32, ttflips=False):
        with tqdm.tqdm(total=len(dataset), unit="files", desc="classification of images from " + str(dataset)) as pbar:
            for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips):
                b=v
                for i in range(len(b.data)):
                    id=b.data[i]
                    cb(id,b.results[i],data)
                pbar.update(batch_size)

    def predict_all_to_array(self, dataset, fold, stage, limit=-1, batch_size=32, ttflips=False):
        res=[]
        with tqdm.tqdm(total=len(dataset), unit="files", desc="classification of images from " + str(dataset)) as pbar:
            for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips):
                b=v
                for i in range(len(b.data)):
                    id=b.data[i]
                    res.append(b.results[i])
                pbar.update(batch_size)
        return np.array(res)

def parse(path) -> GenericPipeline:
    cfg = configloader.parse("generic", path)
    cfg.path = path;
    return cfg
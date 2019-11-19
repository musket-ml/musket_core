import os
import numpy as np
import yaml
import traceback
import csv
import keras
import tqdm
import pandas as pd
from musket_core import model
import typing
import musket_core.image_datasets
import musket_core.splits
import musket_core.tasks as tasks
from musket_core.image_datasets import WithBackgrounds,Backgrounds,CropAndSplit
from musket_core.utils import ensure
from keras.utils import multi_gpu_model
from musket_core.quasymodels import AnsembleModel,BatchCrop
import musket_core.datasets as datasets
from  musket_core import net_declaration as nd
from musket_core import losses, configloader
import keras.optimizers as opt
from musket_core.lr_finder import LRFinder
from musket_core.logger import CSVLogger
from musket_core import utils
import musket_core.multigpu_checkpoint as alt
from musket_core.structure_constants import constructSummaryYamlPath
from keras.callbacks import  LambdaCallback
import keras.backend as K
import imgaug
import keras.utils.data_utils as _du
import copy
from musket_core.clr_callback import CyclicLR,AllLogger
from musket_core.lr_variation_callback import LRVariator
import tensorflow as tf
from musket_core.context import context
keras.callbacks.CyclicLR= CyclicLR
keras.callbacks.LRVariator = LRVariator
from musket_core import predictions
keras.utils.get_custom_objects()["macro_f1"]= musket_core.losses.macro_f1
keras.utils.get_custom_objects()["f1_loss"]= musket_core.losses.f1_loss
keras.utils.get_custom_objects()["dice"]= musket_core.losses.dice
keras.utils.get_custom_objects()["iou"]= musket_core.losses.iou_coef
keras.utils.get_custom_objects()["iou_coef"]= musket_core.losses.iou_coef
keras.utils.get_custom_objects()["iot"]= musket_core.losses.iot_coef
keras.utils.get_custom_objects()["lovasz_loss"]= musket_core.losses.lovasz_loss
keras.utils.get_custom_objects()["iou_loss"]= musket_core.losses.iou_coef_loss
keras.utils.get_custom_objects()["dice_loss"]= musket_core.losses.dice_coef_loss
keras.utils.get_custom_objects()["dice"]= musket_core.losses.dice
keras.utils.get_custom_objects()["jaccard_loss"]= musket_core.losses.jaccard_distance_loss
keras.utils.get_custom_objects()["focal_loss"]= musket_core.losses.focal_loss
keras.utils.get_custom_objects()["l2_loss"]= musket_core.losses.l2_loss
from musket_core.parralel import  Task
keras.utils.get_custom_objects().update({'matthews_correlation': musket_core.losses.matthews_correlation})
keras.utils.get_custom_objects().update({'log_loss': musket_core.losses.log_loss})
from musket_core import net_declaration as net
def relu6(x):
    return K.relu(x, max_value=6)

keras.utils.get_custom_objects()["relu6"]=relu6
from musket_core.datasets import DataSet, MeanDataSet, BufferedWriteableDS, WriteableDataSet

dataset_augmenters={

}
extra_train={}

from  threading import Lock

_patched=False
def patch_concurency_issues():
    global _patched
    if _patched:
        return
    _patched=True
    global tmpMethod, z, _keras_seq_lock
    tmpMethod = tqdm.tqdm._decr_instances

    def replacementMethod(*args, **kwargs):
        try:
            tmpMethod(*args, **kwargs)
        except:
            pass

    z = _du.SequenceEnqueuer.__init__
    _keras_seq_lock = Lock()

    def new_Constructor(*args, **kwargs):
        _keras_seq_lock.acquire()
        try:
            z(*args, **kwargs)
        finally:
            _keras_seq_lock.release()

    _du.SequenceEnqueuer.__init__ = new_Constructor
    tqdm.tqdm._decr_instances = replacementMethod

patch_concurency_issues()

class Rotate90(imgaug.augmenters.Affine):
    def __init__(self, enabled=True):
        if enabled:
            super(Rotate90, self).__init__(rotate=imgaug.parameters.Choice([0, 90, 180, 270]))
        else:
            super(Rotate90, self).__init__(rotate=imgaug.parameters.Choice([0]))

imgaug.augmenters.Rotate90 = Rotate90



def maxEpoch(file):
    if not os.path.exists(file):
        return -1;
    with open(file, 'r') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
         epoch=-1;
         num=0;
         for row in spamreader:
             if num>0:
                epoch=max(epoch,int(row[0])+1)
             num = num + 1;
         return epoch;


class ExecutionConfig:

    def __init__(self, fold=0, stage=0, subsample=1.0, dr: str = "", drawingFunction=None):
        self.subsample = subsample
        self.stage = stage
        self.fold = fold
        self.dirName = dr
        self.drawingFunction = drawingFunction
        pass
    
    def predictions_dump(self,y=False):
        ensure(os.path.join(self.dirName, "predictions"))
        if y:
            return os.path.join(self.dirName, "predictions","validation" + str(self.fold) + "." + str(self.stage) + "-gt.csv")
        else:
            return os.path.join(self.dirName, "predictions","validation" + str(self.fold) + "." + str(self.stage) + "-pt.csv")
        
    def predictions_holdout(self,y=False):
        ensure(os.path.join(self.dirName, "predictions"))
        if y:
            return os.path.join(self.dirName, "predictions","holdout-" + str(self.fold) + "." + str(self.stage) + "-gt.csv")
        else:
            return os.path.join(self.dirName, "predictions","holdout-" + str(self.fold) + "." + str(self.stage) + "-pt.csv")
            

    def weightsPath(self):
        ensure(os.path.join(self.dirName, "weights"))
        return os.path.join(self.dirName, "weights","best-" + str(self.fold) + "." + str(self.stage) + ".weights")

    def last_weightsPath(self):
        ensure(os.path.join(self.dirName, "weights"))
        return os.path.join(self.dirName, "weights","last-" + str(self.fold) + "." + str(self.stage) + ".weights")

    def classifier_weightsPath(self):
        ensure(os.path.join(self.dirName, "classify_weights"))
        return os.path.join(self.dirName, "classify_weights","best-" + str(self.fold) + "." + str(self.stage) + ".weights")

    def metricsPath(self):
        ensure(os.path.join(self.dirName, "metrics"))
        return os.path.join(self.dirName, "metrics","metrics-" + str(self.fold) + "." + str(self.stage) + ".csv")

    def classifier_metricsPath(self):
        ensure(os.path.join(self.dirName, "classify_metrics"))
        return os.path.join(self.dirName, "classify_metrics","metrics-" + str(self.fold) + "." + str(self.stage) + ".csv")


def ansemblePredictions(sourceFolder, folders:[str], cb, data, weights=None):
    if weights==None:
        weights=[]
        for f in folders:
            weights.append(1.0)
    for i in os.listdir(sourceFolder):
       a=None
       num = 0
       sw = 0
       for f in folders:
           sw=sw+weights[num]
           if a is None:
            a=np.load(f+i[0:i.index(".")]+".npy")*weights[num]
           else:
            a=a+np.load(f+i[0:i.index(".")]+".npy")*weights[num]
           num=num+1
       a=a/sw
       cb(i, a, data)

def dir_list(spath):
    if isinstance(spath, datasets.ConstrainedDirectory):
        return spath.filters
    return os.listdir(spath)
def copy_if_exist(name: str, fr: dict, trg: dict):
    if name in fr:
        trg[name] = fr[name]

def create_with(names: [str], fr: dict):
    res = {}
    for v in names:
        copy_if_exist(v, fr, res)
    return res;

class ScoreAndTreshold:

    def __init__(self,score,treshold):
        self.score=score
        self.treshold=treshold


    def __str__(self):
        return "score:"+str(self.score)+": treshold:"+str(self.treshold)

def threshold_search(predsDS:DataSet, func, batch_size:int = -1):

    if isinstance(func,str):
        func_=keras.metrics.get(func)
        def wrapped(x,y):

            v1= K.constant(x)
            v2 = K.constant(y)
            return K.eval(func_(v1,v2))
        func=wrapped
    best_threshold = 0
    best_score = 0
    K.clear_session()
    for threshold in tqdm.tqdm([i * 0.01 for i in range(100)]):

        def func1(y_true, y_proba):
            return func(y_true.astype(np.float64), (y_proba > threshold))

        score = applyFunctionToDS(predsDS, func1, batch_size)
        if np.mean(score) > np.mean(best_score):
            best_threshold = threshold
            best_score = score
    return ScoreAndTreshold(best_score,best_threshold)

def need_threshold(func:str)->bool:
    func_ = keras.metrics.get(func)
    if func_ is not None and hasattr(func_, "need_threshold") and func_.need_threshold == False:
        return False
    return True


def eval_metric(predsDS:DataSet, func, batch_size:int = -1):

    if isinstance(func,str):
        func_=keras.metrics.get(func)
        if (func=="binary_accuracy"):
            func=losses.binary_accuracy_numpy
        elif (func=="iou_coef"):
            func=losses.iou_numpy_true_negative_is_one
        elif (func=="dice"):
            func=losses.dice_numpy
        else:    
            def wrapped(x,y):
                with tf.Session().as_default():
                    with tf.device("/cpu:0"):
                        v1= K.constant(x)
                        v2 = K.constant(y)
                        return K.eval(func_(v1,v2))
            func=wrapped

    result = applyFunctionToDS(predsDS, func, batch_size)
    return result


def applyFunctionToDS(dsWithPredictions, func, batch_size:int):

    l = len(dsWithPredictions)
    if batch_size < 0:
        batch_size = l

    resultArr = []
    resultScalarArr = []
    for ind in tqdm.tqdm(range(0, l, batch_size)):
        end = min(ind + batch_size, l)
        items = dsWithPredictions[ind:end]
        y_true = np.array([i.y for i in items])
        y_proba = np.array([i.prediction for i in items])
        #y_true = y_true.reshape(y_proba.shape)
        if y_true.dtype == np.bool:
            y_true = y_true.astype(np.float32)
        if y_proba.dtype == np.bool:
            y_proba = y_proba.astype(np.float32)
        batch_value = func(y_true, y_proba)
        if isinstance(batch_value, np.ndarray):
            resultArr.append(batch_value.mean())
        else:
            resultScalarArr.append(batch_value)

    if len(resultArr) > 0:
        if len(resultArr)==1:
            return resultArr[0] 
        return np.concatenate(resultArr)
    else:
        return np.array(resultScalarArr)


class FoldWork:

    def __init__(self,cfg,num,start_from_stage,subsample,drawingFunction,folds):
        self.cfg=cfg
        self.num=num
        self.start_from_stage=start_from_stage
        self.subsample=subsample
        self.drawingFunction=drawingFunction
        self.folds=folds

    def __call__(self, *args, **kwargs):
        model = self.cfg.createAndCompile()
        cb=self.cfg.callbacks
        for s in range(0, len(self.cfg.stages)):
            if s < self.start_from_stage:
                self.cfg.skip_stage(self.num, model, s, self.subsample)
                continue
            st: Stage = self.cfg.stages[s]
            ec = ExecutionConfig(fold=self.num, stage=s, subsample=self.subsample, dr=self.cfg.directory(),
                                 drawingFunction=self.drawingFunction)
            st.execute(self.folds, model, ec,copy.deepcopy(cb))

class ReportWork:
    def __init__(self,cfg,foldsToExecute,subsample):
        self.cfg=cfg
        self.foldsToExecute=foldsToExecute
        self.subsample=subsample

    def __call__(self, *args, **kwargs):
        self.cfg.generateReports(self.foldsToExecute, self.subsample)


class GenericTaskConfig(model.IGenericTaskConfig):

    def __init__(self,**atrs):
        self.batch = 8
        self.holdoutArr=None
        self.all = atrs
        self.groupFunc=None
        self.imports=[]
        self.datasets_path=None
        self._dataset=None
        self.dumpPredictionsToCSV=False
        self._reporter=None
        self.testTimeAugmentation=None
        self.stratified=False
        self.needsSessionForPrediction=True
        self.compressPredictionsAsInts=True
        self.preprocessing=None
        self.verbose = 1
        self.compressScale=None
        self._projectDir=None
        self.manualResize=None
        self.separatePredictions=True
        self.dataset=None
        self.noTrain = False
        self.inference_batch=32
        self.saveLast = False
        self.folds_count = 5
        self.add_to_train = None
        self.random_state = 33
        self.primary_metric = "val_binary_accuracy"
        self.primary_metric_mode = "auto"
        self.stages = []
        self.shape=None
        self.gpus = 1
        self.lr = 0.001
        self.callbacks = []
        self.declarations={}
        self.optimizer = None
        self.loss = None
        self.testSplit = 0
        self.validationSplit = 0.2
        self.pretrain_transform=None
        self.testSplitSeed = 123
        self.path = None
        self.metrics = []
        self.final_metrics=[]
        self.experiment_result=None
        self.resume = False
        self.weights = None
        self.transforms=[]
        self.augmentation=[]
        self.extra_train_data=None
        self.dataset_augmenter=None
        self.datasets=None
        self.architecture=None
        self.encoder_weights = None
        self.activation = None
        self.bgr = None
        self.rate = 0.5
        self.showDataExamples = False
        self.crops = None
        self.flipPred = True
        self.copyWeights = False
        self.maxEpochSize = None
        self.dropout = 0
        self.dataset_clazz = datasets.DefaultKFoldedDataSet

        self.canceled_by_timer = False

        for v in atrs:
            val = atrs[v]
            val = self._update_from_config(v, val)
            setattr(self, v, val)
        pass
        if isinstance(self.metrics,str):
            self.metrics=[self.metrics]
        if isinstance(self.final_metrics, str):
            self.final_metrics = [self.final_metrics]

    def _update_from_config(self, v, val):
        if v == 'callbacks':
            cs = []
            val = configloader.parse("callbacks", val)
            if val is not None:
                val = val + cs
        if v == 'stages':
            val = [self.createStage(x) for x in val]
        return val

    def inject_task_specific_transforms(self, ds, transforms):
        return ds

    def setHoldout(self, arr):
        self.holdoutArr = arr

    def doGetHoldout(self, ds):
        ho = self.holdoutArr
        if ho is None and hasattr(ds,'holdoutArr'):
            ho = getattr(ds,'holdoutArr')
        if ho is not None:
            train = list(set(range(0,len(ds)-1)).difference(ho))

            return datasets.SubDataSet(ds, train), datasets.SubDataSet(ds, ho)

        else:
            return musket_core.splits.split(ds, self.testSplit, self.testSplitSeed, self.stratified, self.groupFunc)


    def holdout(self, ds=None):
        if ds is None:
            ds=self.get_dataset()
        if os.path.exists(self.path + ".holdout_split"):
            trI,hI = utils.load_yaml(self.path + ".holdout_split")
            train=datasets.SubDataSet(ds,trI)
            test = datasets.SubDataSet(ds,hI)
            return test
        if self.testSplit>0 or self.holdoutArr is not None:
            train,test=self.doGetHoldout(ds)
            return test
            pass
        raise ValueError("This configuration does not have holdout")
    
    def train_without_holdout(self, ds=None):
        if ds is None:
            ds=self.get_dataset()
        if os.path.exists(self.path + ".holdout_split"):
            trI,hI = utils.load_yaml(self.path + ".holdout_split")
            train=datasets.SubDataSet(ds,trI)
            test = datasets.SubDataSet(ds,hI)
            return train
        if self.testSplit>0 or self.holdoutArr is not None:
            train,test=self.doGetHoldout(ds)
            return train
            pass
        raise ValueError("This configuration does not have holdout")

    def kfold(self, ds=None, indeces=None,batch=None)-> datasets.DefaultKFoldedDataSet:
        if ds is None:
            ds=self.get_dataset()
        inputFolds = None
        if hasattr(ds,'folds'):
            inputFolds = getattr(ds,'folds')
        if self.testSplit>0 or hasattr(ds, "holdoutArr"):
            if os.path.exists(self.path + ".holdout_split"):
                trI,hI = utils.load_yaml(self.path + ".holdout_split")
                train=datasets.SubDataSet(ds,trI)
                test = datasets.SubDataSet(ds,hI)
            else:
                train,test=self.doGetHoldout(ds)
                utils.save_yaml(self.path + ".holdout_split",(train.indexes,test.indexes))
            ds=train
            pass
        if self.pretrain_transform is not None:
            ds=nd.create_dataset_transformer(self.pretrain_transform,self.imports)(ds)
        if batch is None:
            batch=self.batch
        if indeces is None: indeces=range(0,len(ds))
        transforms = [] + self.transforms
        ds = self.inject_task_specific_transforms(ds, transforms)
        split_loaded=False
        if inputFolds is not None:
            ds.folds = inputFolds
        elif os.path.exists(self.path+".folds_split"):
            folds=utils.load_yaml(self.path+".folds_split")
            split_loaded=True
            ds.folds=folds
        kf= self.dataset_clazz(ds, indeces, self.augmentation, transforms, batchSize=batch,rs=self.random_state,folds=self.folds_count,stratified=self.stratified,groupFunc=self.groupFunc,validationSplit=self.validationSplit,maxEpochSize=self.maxEpochSize)
        if not split_loaded:
            kf.save(self.path+".folds_split")
        if self.noTrain:
            kf.clear_train()
        if self.extra_train_data is not None:
            if str(self.extra_train_data) in extra_train:
                kf.addToTrain(extra_train[self.extra_train_data])
            else:
                if isinstance(self.extra_train_data, str):
                    fw = self.datasets[self.extra_train_data]
                else:
                    fw=self.extra_train_data
                dataset = net.create_dataset_from_config(self.declarations, fw, self.imports)
                if self.preprocessing is not None and self.preprocessing != "":
                    dataset.cfg=self
                    dataset = net.create_preprocessor_from_config(self.declarations, dataset, self.preprocessing,
                                                                  self.imports)
                kf.addToTrain(dataset)
                
            
        if self.dataset_augmenter is not None:
            args = dict(self.dataset_augmenter)
            del args["name"]
            ag=dataset_augmenters[self.dataset_augmenter["name"]](**args)
            kf=ag(kf)
            pass
        return kf

    def predict_on_dataset(self, dataset, fold=0, stage=0, limit=-1, batch_size=32, ttflips=False, cacheModel=False):
        raise ValueError("Not implemented")

    def predict_all_to_array(self, dataset, fold=None, stage=None, limit=-1, batch_size=None, ttflips=False):
        if batch_size is None:
            batch_size=self.inference_batch
        if fold is None:
            fold=list(range(self.folds_count))
        if stage is None:
            stage=list(range(len(self.stages)))
        if isinstance(dataset,str):
            dataset=self.get_dataset(dataset)
        res=[]
        with tqdm.tqdm(total=len(dataset), unit="files", desc="prediction from  " + str(dataset)) as pbar:
            for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips):
                b=v
                
                #fixme: b.data is empty
                for i in range(len(b.data)):
                    res.append(b.results[i])
                pbar.update(batch_size)
        return np.array(res)


    def load_writeable_dataset(self, ds, path)->DataSet:
        rr = np.load(path)
        resName = (ds.name if hasattr(ds, "name") else "") + "_predictions"
        result = BufferedWriteableDS(ds, resName, path, rr)
        return result

    def create_writeable_dataset(self, dataset:DataSet, dsPath:str)->WriteableDataSet:
        resName = (dataset.name if hasattr(dataset, "name") else "") + "_predictions"
        result = BufferedWriteableDS(dataset, resName, dsPath)
        return result
    
    def isMultiOutput(self):
        return isinstance(self.classes,list);

    def predict_all_to_dataset(self, dataset, fold=None, stage=None, limit=-1, batch_size=None, ttflips=False, dsPath = None, cacheModel=False,verbose=1)->DataSet:

        result = self.create_writeable_dataset(dataset, dsPath)

        if batch_size is None:
            batch_size=self.inference_batch
        if fold is None:
            fold=list(range(self.folds_count))
        if stage is None:
            stage=list(range(len(self.stages)))
        if isinstance(dataset,str):
            dataset=self.get_dataset(dataset)
        
            
        if verbose>0:    
            with tqdm.tqdm(total=len(dataset), unit="files", desc="prediction from  " + str(dataset)) as pbar:
                for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips, cacheModel=cacheModel):
                    b=v
                    for i in range(len(b.results)):
                        result.append(b.results[i])
                    pbar.update(batch_size)
            result.commit()        
        else:
            for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips, cacheModel=cacheModel):
                for i in range(len(v.results)):
                    result.append(v.results[i])           
        
        return result



    def _folds(self):
        res=[]
        for i in range(self.folds_count):
            if os.path.exists(os.path.join(os.path.dirname(self.path),ExecutionConfig(i).weightsPath())):
                res.append(i)
        return res        

    def predictions(self,name,fold=None,stage=None)->WriteableDataSet:
        if fold is None:
            fold=self._folds()
            if "validation" in name and name.index("validation")==0:
                return self.predictions("validation", int(name[-1:]), stage)
        if stage is None:
            stage=list(range(len(self.stages)))    
        return musket_core.predictions.get_predictions(self,name,fold,stage)

    def find_treshold(self,ds,fold,func,stage=0):

        if isinstance(stage,list) or isinstance(stage,tuple):
            pa = []
            for i in stage:
                pa.append(predictions.Prediction(self, fold, i, None, ds).calculate())
            predictedDS = MeanDataSet(pa)
        else: predictedDS = predictions.Prediction(self, fold, stage, None, ds).calculate()
        return threshold_search(predictedDS,func,self.get_eval_batch())

    def find_optimal_treshold_by_validation(self,func,stages=None):
        tresh = []
        for fold in range(self.folds_count):
            predsDS = predictions.get_validation_prediction(self,fold,stages)
            tr = threshold_search(predsDS, func, self.get_eval_batch())
            tresh.append(tr.treshold)
        tr = np.mean(np.array(tresh))
        return tr
    

    def find_optimal_treshold_by_holdout(self,func,stages=None):
        predsDS=predictions.get_holdout_prediction(self,None,stages)
        return threshold_search(predsDS, func, self.get_eval_batch()).treshold


    def createAndCompile(self, lr=None, loss=None)->keras.Model:
        context.projectPath=self.get_project_path()
        return self.compile(self.createNet(), self.createOptimizer(lr=lr), loss=loss)

    def validate(self):
        model = self.createAndCompile()
        model.summary()
        
    def createNetForInference(self,fold=0,stage=-1):
        if stage == -1: stage = len(self.stages) - 1
        if isinstance(fold,list):
            mdl=[]
            for i in fold:
                ec = ExecutionConfig(i, stage, subsample=1.0, dr=self.directory())
                if os.path.exists(ec.weightsPath()) or isinstance(stage, list):
                    rs=self.createNetForInference(i,stage);
                    if rs is not None:
                        mdl.append(rs)
                else:
                    print("can not find weights for stage:"+str(stage)+" fold: "+str(i)+" skipping.")
            if len(mdl)==0:
                return None
            if len(mdl)==1:
                return mdl[0]            
            return AnsembleModel(mdl)
        if isinstance(stage,list):
            mdl=[]
            for s in stage:
                ec = ExecutionConfig(fold, s, subsample=1.0, dr=self.directory())
                if os.path.exists(ec.weightsPath()) or isinstance(fold, list):
                    rs=self.createNetForInference(fold,s)
                    if rs is not None:
                        mdl.append(rs)
                else:
                    print("can not find weights for stage:"+str(s)+" fold: "+str(fold)+" skipping.")
            if len(mdl)==0:
                return None
            if len(mdl)==1:
                return mdl[0]           
            return AnsembleModel(mdl)
        
        ec = ExecutionConfig(fold=fold, stage=stage, subsample=1.0, dr=self.directory())
        context.train_mode=False
        try:
            if os.path.exists(self.path+".ncx"):
                context.net_cx=utils.load(self.path+".ncx")
            model=self.createNet()
            if not os.path.exists(ec.weightsPath()):
                return None
            model.load_weights(ec.weightsPath(),False)
            return model
        finally:
            context.train_mode=True  
            context.net_cx=[]  

    def createNet(self):
        raise ValueError("Not implemented")

    def validation(self,ds=None,foldNum=0):
        if ds is None:
            ds = self.get_dataset()
        if isinstance(ds,int):
            foldNum=ds
            ds=self.get_dataset()
        if isinstance(foldNum, list):
            foldNum=foldNum[0]
        if self.testSplit>0:
            ds=self.train_without_holdout()    
        ids=self.kfold(ds).indexes(foldNum,False)
        r=datasets.SubDataSet(ds, ids)
        r.name="validation"+str(foldNum)
        return r

    def train(self,ds,foldNum):
        ids=self.kfold(ds).indexes(foldNum,True)
        if self.testSplit>0:
            ds=self.train_without_holdout()  
        r=datasets.SubDataSet(ds,ids)
        r.name="train"+str(foldNum)
        return r

    def createOptimizer(self, lr=None):
        if not self.optimizer:
            return None

        r = getattr(opt, self.optimizer)
        ds = create_with(["lr", "clipnorm", "clipvalue"], self.all)
        if lr:
            ds["lr"] = lr
        return r(**ds)

    def skip_stage(self, i, model, s, subsample):
        st: Stage = self.stages[s]
        ec = ExecutionConfig(fold=i, stage=s, subsample=subsample, dr=self.directory())
        if os.path.exists(ec.weightsPath()):
            model.load_weights(ec.weightsPath())
            if 'unfreeze_encoder' in st.dict and st.dict['unfreeze_encoder']:
                st.unfreeze(model)

    def createStage(self,x):
        return Stage(x,self)

    def lr_find(self, d, foldsToExecute=None,stage=0,subsample=1.0,start_lr=0.000001,end_lr=1.0,epochs=5):
        dn = self.directory()
        if os.path.exists(os.path.join(dn, "3summary.yaml")):
            raise ValueError("Experiment is already finished!!!!")
        folds = self.kfold(d)

        for i in range(len(folds.folds)):
            if foldsToExecute:
                if not i in foldsToExecute:
                    continue
            model = self.createAndCompile()
            for s in range(0, len(self.stages)):
                if s<stage:
                    self.skip_stage(i, model, s, subsample)
                    continue
                st: Stage = self.stages[s]
                ec = ExecutionConfig(fold=i, stage=s, subsample=subsample, dr=self.directory())
                return st.lr_find(folds, model, ec,start_lr,end_lr,epochs)


    def setAllowResume(self,resume):
        self.resume=resume

    def compile(self, net: keras.Model, opt: keras.optimizers.Optimizer, loss:str=None)->keras.Model:
        if loss==None:
            loss=self.loss
        if "+" in loss:
            loss= losses.composite_loss(loss)
        if loss=='lovasz_loss' and isinstance(net.layers[-1],keras.layers.Activation):
            net=keras.Model(net.layers[0].input,net.layers[-1].input)
        if loss:
            net.compile(opt, loss, self.metrics)
        else:
            net.compile(opt, self.loss, self.metrics)
        return net

    def load_model(self, fold: int = 0, stage: int = -1):
        if isinstance(fold,list):
            mdl=[]
            for i in fold:
                mdl.append(self.load_model(i,stage))
            return AnsembleModel(mdl)
        if isinstance(stage,list):
            mdl=[]
            for s in stage:
                mdl.append(self.load_model(fold,s))
            return AnsembleModel(mdl)
        if stage == -1: stage = len(self.stages) - 1
        ec = ExecutionConfig(fold=fold, stage=stage, subsample=1.0, dr=self.directory())
        model = self.createAndCompile()
        model.load_weights(ec.weightsPath(),False)
        return model

    def info(self,metric=None):
        if metric is None:
            metric=self.primary_metric
        ln=self.folds_count
        res=[]
        for i in range(ln):
            for s in range(0, len(self.stages)):
                st: Stage = self.stages[s]
                ec = ExecutionConfig(fold=i, stage=s, dr=self.directory())
                if os.path.exists(ec.metricsPath()):
                    try:
                        fr=pd.read_csv(ec.metricsPath())
                        res.append(TaskConfigInfo(i,s,fr[metric].max(),fr["lr"].min()))
                    except:
                        pass
        return res

    def directory(self):
        return os.path.dirname(self.path)

    def get_dataset(self,dataSetName=None):
        if dataSetName is not None:
            if dataSetName=="holdout":
                return self.holdout(self.get_dataset())
            if dataSetName=="validation":
                return self.validation(self.get_dataset(),0)
            if dataSetName=="train":
                return self.train(self.get_dataset(),0)
            return self.parse_dataset(dataSetName)


        if self._dataset is not None:
            return self._dataset
        if self.dataset is not None:
            self._dataset=self.parse_dataset()
            if hasattr(self._dataset, 'folds'):
                self.folds_count = len(self._dataset.folds)
            return self._dataset
        raise ValueError("Data set is not defined for this config")

    def fit(self, dataset_ = None, subsample=1.0, foldsToExecute=None, start_from_stage=0, drawingFunction = None,parallel=False)->typing.Collection[Task]:
        if dataset_ is None:
            dataset = self.get_dataset()
        else: dataset=dataset_

        dataset = self._adapt_before_fit(dataset)
        self._dataset=dataset

        dn = self.directory()
        if os.path.exists(constructSummaryYamlPath(dn)):
            raise ValueError("Experiment is already finished!!!!")
        folds = self.kfold(dataset, None)
        units_of_works=[]
        for i in range(len(folds.folds)):
            if foldsToExecute:
                if not i in foldsToExecute:
                    continue
            fw=FoldWork(self,i,start_from_stage,subsample,drawingFunction,folds)
            if not parallel:
                fw()
            else:
                units_of_works.append(fw)
        if self._reporter is not None and self._reporter.isCanceled():
            return []
        if parallel:
            units_of_works=[Task(x) for x in units_of_works]
            rw=Task(ReportWork(self,foldsToExecute,subsample))
            rw.deps=units_of_works.copy()
            units_of_works.append(rw)
            return units_of_works
        else: self.generateReports(foldsToExecute, subsample)



    def generateReports(self, foldsToExecute=None, subsample=1.0):
        if self.canceled_by_timer:
            return

        dn = self.directory()
        with open(constructSummaryYamlPath(dn), "w") as f:
            fw=foldsToExecute
            if fw is None:
                fw=list(range(self.folds_count))
            initial = {"completed": True, "cfgName": os.path.basename(self.path), "subsample": subsample,
                       "folds": fw}
            metrics = self.createSummary(foldsToExecute,subsample)
            for k in metrics:
                initial[k] = metrics[k]
            yaml.dump(
                initial,
                f)


    def _append_metric(self, foldsToExecute, ms, m, s):
        def isStat(d):
            return "min" in d and "max" in d and "std" in d and "mean" in d
        mv = predictions.cross_validation_stat(self, m, s, folds=foldsToExecute)
        if isinstance(mv, dict) and not isStat(mv):
            
            for k in mv:
                ms[k] = mv[k]
        else:
            
            ms[m] = mv
        if self.hasHoldout():
            mv = predictions.holdout_stat(self, m, s)
            if isinstance(mv, dict) and not isStat(mv):
                for k in mv:
                    ms[k + "_holdout"] = mv[k]
            
            else:
                ms[m + "_holdout"] = mv
        


    def hasHoldout(self):
        return self.testSplit > 0 or hasattr(self._dataset, "holdoutArr")

    def createSummary(self,foldsToExecute, subsample):
        stagesStat=[]
        all_metrics=self.metrics+self.final_metrics
        if self.experiment_result is not None:
            if not self.experiment_result in all_metrics:
                all_metrics.append(self.experiment_result)
        
        metric = self.primary_metric        
        if "val_" in metric:
            metric=metric[4:]
        for stage in range(len(self.stages)):
            ms={}
            if self._reporter is not None and self._reporter.isCanceled():
                return {"canceled": True }
            for m in all_metrics:
                s = [stage]
                self._append_metric(foldsToExecute, ms, m, s)
                        
                                       
            stagesStat.append(ms)
        all={}
        s=None;
        for m in all_metrics:
            if self._reporter is not None and self._reporter.isCanceled():
                return {"canceled": True }
            self._append_metric(foldsToExecute, all, m, s)
        if self.dumpPredictionsToCSV:
            for i in self._folds() :
                    vl=self.predictions("validation", i, len(self.stages)-1)
                    c=ExecutionConfig(i,len(self.stages)-1,dr=os.path.dirname(self.path))
                    vl.dump(c.predictions_dump(True), 0.5, encode_y=True)
                    vl.dump(c.predictions_dump(False), 0.5, encode_y=False)                        
            if self.hasHoldout():
                c=ExecutionConfig(self._folds(),len(self.stages)-1,dr=os.path.dirname(self.path))
                vl=self.predictions("holdout", self._folds(), len(self.stages)-1)
                vl.dump(c.predictions_holdout(True), 0.5, True)
                vl.dump(c.predictions_holdout(False), 0.5, False)        
        return {"stages":stagesStat,"allStages":all}

    def _adapt_before_fit(self, dataset):
        return dataset

    def eval_tasks(self):
        path = os.path.dirname(os.path.abspath(self.path))

        callbacks = tasks.load_task_sets(path, self.import_tasks)

        ds_config = self.pickup_ds_config()

        run_tasks = self.run_tasks

        tasks_set_id = 0

        for item in run_tasks:
            print("task set: " + str(tasks_set_id))

            tasks_set = item["tasks"]

            dataset_id = item["dataset"]

            fold = item["fold"]
            stage = item["stage"]

            dataset = musket_core.image_datasets.DS_Wrapper(dataset_id, ds_config, self.path)

            with tqdm.tqdm(total=len(dataset)) as pbar:
                self.eval_task_set(item, tasks_set, tasks_set_id, path, dataset, fold, stage, callbacks, pbar)

            tasks_set_id += 1

    def eval_task_set(self, task_item, tasks_set, tasks_set_id, path, dataset, fold, stage, callbacks, progress_bar):
        task_runners = {}

        for ts in tasks_set:
            print("\t" + ts)

            task_runners[ts] = tasks.create_task_runner(ts, tasks_set_id, task_item.get("parameters", {}), path, callbacks)

        print()

        for predictions in self.predict_on_dataset(dataset, fold, stage, batch_size=self.batch):
            count = 0

            for id in predictions.data:
                ds_item = dataset.item_by_id(id)

                ds_item.p_y = predictions.segmentation_maps_aug[count].arr
                ds_item.p_x = predictions.images_aug[count]

                for task_name in tasks_set:
                    tasks.eval_task_for_item(ds_item, task_name, task_runners)

                count += 1

                progress_bar.update(1)

        for runner in task_runners.values():
            runner.end()

    def get_default_dataset_folder(self):
        if self.datasets_path is not None:
            return self.datasets_path
        return os.path.join(self.get_project_path(),"data")



    def get_project_path(self):
        if self._projectDir is not None:
            return self._projectDir
        v=self.path
        while v is not None and len(v)>0:
            v=os.path.dirname(v)
            if os.path.exists(os.path.join(v,"modules")):
                self._projectDir=v
                return v
        self._projectDir=os.path.dirname(self.path)
        self._projectDir=os.path.dirname(self._projectDir)
        print(self._projectDir)
        return self._projectDir

    def parse_dataset(self,datasetName=None):
        #try:
            context.projectPath=self.get_project_path()
            fw = self.dataset
            if self.datasets_path is not None:
                os.chdir(self.get_default_dataset_folder())  # TODO review
            if datasetName is not None:
                fw = self.datasets[datasetName]
            if isinstance(fw, str):
                fw = self.datasets[datasetName]
            if self.dataset is not None:
                dataset = net.create_dataset_from_config(self.declarations, fw, self.imports)
                
                if self.preprocessing is not None and self.preprocessing != "":
                    dataset.cfg=self
                    dataset = net.create_preprocessor_from_config(self.declarations, dataset, self.preprocessing,
                                                                  self.imports)
                return dataset
            return None
#         except:
#             ds_config = self.pickup_ds_config()
#             return musket_core.image_datasets.DS_Wrapper(self.dataset, ds_config, self.path)


    def pickup_ds_config(self):
        return self.datasets

    def clean(self, cleaned):
        cleaned.pop("datasets", None)
        cleaned.pop("dataset", None)
        cleaned.pop("run_tasks", None)
        cleaned.pop("import_tasks", None)
        cleaned.pop("maxEpochSize", None)

class TaskConfigInfo:

    def __init__(self,fold,stage,best,lr):
        self.fold = fold
        self.stage = stage
        self.best = best
        self.lr = lr


import cv2
class GenericImageTaskConfig(GenericTaskConfig):

    def __init__(self,**atrs):
        super().__init__(**atrs)
        self.mdl = None

    def _update_from_config(self, v, val):
        if v == 'augmentation' and val is not None:
            if "BackgroundReplacer" in val:
                bgr = val["BackgroundReplacer"]
                aug = None
                erosion = 0
                if "erosion" in bgr:
                    erosion = bgr["erosion"]
                if "augmenters" in bgr:
                    aug = bgr["augmenters"]
                    aug = configloader.parse("augmenters", aug)
                    aug = imgaug.augmenters.Sequential(aug)
                self.bgr = Backgrounds(bgr["path"], erosion=erosion, augmenters=aug)
                self.bgr.rate = bgr["rate"]
                del val["BackgroundReplacer"]
            val = configloader.parse("augmenters", val)
        if v == 'transforms':
            val = configloader.parse("augmenters", val)
        if v == 'callbacks':
            cs = []
            val = configloader.parse("callbacks", val)
            if val is not None:
                val = val + cs
        if v == 'stages':
            val = [self.createStage(x) for x in val]
        return val

    def predict_on_directory(self, path, fold=0, stage=0, limit=-1, batch_size=None, ttflips=False):
        return self.predict_on_dataset(datasets.DirectoryDataSet(path), fold, stage, limit, batch_size, ttflips)

    def predict_on_dataset(self, dataset, fold=0, stage=0, limit=-1, batch_size=None, ttflips=False, cacheModel=False):
        if self.testTimeAugmentation is not None:
            ttflips=self.testTimeAugmentation
        if batch_size is None:
            batch_size=self.inference_batch

        if cacheModel:
            if self.mdl is None:
                self.mdl = self.createNetForInference(fold, stage)
            mdl = self.mdl
        else:
            mdl = self.createNetForInference(fold, stage)

        if self.crops is not None:
            mdl=BatchCrop(self.crops,mdl)
        ta = self.transformAugmentor()
        for original_batch in datasets.batch_generator(dataset, batch_size, limit):
            for batch in ta.augment_batches([original_batch]):
                res = self.predict_on_batch(mdl, ttflips, batch)
                resList = [x for x in res]
                for ind in range(len(resList)):
                    img = resList[ind]
                    # FIXME
                    
                    if not self.manualResize and self.flipPred:
                        unaug = original_batch.images[ind]
                        restored = imgaug.imresize_single_image(img,(unaug.shape[0],unaug.shape[1]),cv2.INTER_AREA)
                    else:
                        restored=img    
                    resList[ind] = restored
                self.update(batch,resList)
                batch.results=resList
                yield batch



    def _adapt_before_fit(self, dataset):
        if self.crops is not None:
            dataset = CropAndSplit(dataset, self.crops)
        return dataset

    def update(self,batch,res):
        pass


    def _tr_multioutput_if_needed(self, res):
        if self.isMultiOutput():
            result = []
            for i in range(len(res[0])):
                elementOutputs = []
                for x in res:
                    elementOutputs.append(x[i])
                
                result.append(elementOutputs)
            
            res = result
        return  res

    def predict_on_batch(self, mdl, ttflips, batch):
        o1 = np.array(batch.images_aug)
        res = self._tr_multioutput_if_needed(mdl.predict(o1))
          
        if ttflips == "Horizontal":
            another = imgaug.augmenters.Fliplr(1.0).augment_images(batch.images_aug)
            res1 = self._tr_multioutput_if_needed(mdl.predict(np.array(another)))
            if self.flipPred:
                res1 = imgaug.augmenters.Fliplr(1.0).augment_images(res1)
            res = self._ave([res,res1])
        elif ttflips == "Horizontal_and_vertical":
            s=imgaug.augmenters.Sequential([imgaug.augmenters.Flipud(1.0),imgaug.augmenters.Flipud(1.0)])
            r0 = self.predict_there_and_back(mdl, imgaug.augmenters.Fliplr(1.0), imgaug.augmenters.Fliplr(1.0), batch.images_aug)
            r1 = self.predict_there_and_back(mdl, imgaug.augmenters.Flipud(1.0), imgaug.augmenters.Flipud(1.0), batch.images_aug)
            r2 = self.predict_there_and_back(mdl, s, s, batch.images_aug)    
            res = self._ave([res, r0,r1,r2])    
        elif ttflips:
            res = self.predict_with_all_augs(mdl, ttflips, batch)
            
        return res
    
    def _ave(self,res):
        return np.sum(res,axis=0)/len(res) 

    def predict_with_all_augs(self, mdl, ttflips, batch):
        input_left = batch.images_aug
        input_right = imgaug.augmenters.Fliplr(1.0).augment_images(batch.images_aug)

        out_left = self.predict_with_all_rot_augs(mdl, ttflips,  input_left)
        out_right = self.predict_with_all_rot_augs(mdl, ttflips,  input_right)

        if self.flipPred:
            out_right = imgaug.augmenters.Fliplr(1.0).augment_images(out_right)

        return self._ave([out_left, out_right])

    def predict_with_all_rot_augs(self, mdl, ttflips,  inp):
        rot_90 = imgaug.augmenters.Affine(rotate=90.0)
        rot_180 = imgaug.augmenters.Affine(rotate=180.0)
        rot_270 = imgaug.augmenters.Affine(rotate=270.0)
        

        res_0 = mdl.predict(np.array(inp))
 
        res_180 = self.predict_there_and_back(mdl, rot_180, rot_180, inp)

        res_270 = res_0;
        res_90 = res_180;

        if ttflips:
            

            res_270 = self.predict_there_and_back(mdl, rot_270, rot_90, inp)
            res_90 = self.predict_there_and_back(mdl, rot_90, rot_270, inp)

        return self._ave([res_0 ,res_90 ,res_180 , res_270])

    def predict_there_and_back(self, mdl, there, back, inp):
        augmented_input = there.augment_images(inp)
        there_res = self._tr_multioutput_if_needed(mdl.predict(np.array(augmented_input)))
        if self.flipPred:
            return back.augment_images(there_res)
        return there_res

    def inject_task_specific_transforms(self, ds, transforms):
        if not self.manualResize:
            transforms.append(imgaug.augmenters.Scale({"height": self.shape[0], "width": self.shape[1]}))
        if self.bgr is not None:
            ds = WithBackgrounds(ds, self.bgr)
        return ds

    def predict_on_directory_with_model(self, mdl, path, limit=-1, batch_size=None, ttflips=False):
        if batch_size is None:
            batch_size=self.inference_batch
        ta = self.transformAugmentor()
        with tqdm.tqdm(total=len(dir_list(path)), unit="files", desc="classifying positive  images from " + path) as pbar:
            for v in datasets.batch_generator(datasets.DirectoryDataSet(path), batch_size, limit):
                for z in ta.augment_batches([v]):
                    res = self.predict_on_batch(mdl,ttflips,z)
                    resList = [x for x in res]
                    for ind in range(len(resList)):
                        img = resList[ind]
                        unaug = z.images_unaug[ind]
                        resize = imgaug.augmenters.Scale({"height": unaug.shape[0], "width": unaug.shape[1]})
                        restored = resize.augment_image(img)
                        resList[ind] = restored
                    z.predictions = resList;
                    pbar.update(batch_size)
                    yield z
                    

    def transformAugmentor(self):
        transforms = [] + self.transforms
        if not self.manualResize:
            transforms.append(imgaug.augmenters.Resize({"height": self.shape[0], "width": self.shape[1]}))
        return imgaug.augmenters.Sequential(transforms)

    def adaptNet(self, model, model1, copy=False,sh=4):
        notUpdated = True
        for i in range(0, len(model1.layers)):
            if isinstance(model.layers[i], keras.layers.BatchNormalization) and notUpdated:
                uw = []
                for w in model1.layers[i].get_weights():
                    val = w
                    vvv = np.zeros(shape=(sh), dtype=np.float32)
                    if sh>1:
                        vvv[0:sh-1] = val
                        vvv[sh-1] = (val[0] + val[1] + val[2]) / 3
                    else:
                       vvv[sh-1] = (val[0] + val[1] + val[2]) / 3    
                    uw.append(vvv)
                model.layers[i].set_weights(uw)

            elif isinstance(model.layers[i], keras.layers.Conv2D) and notUpdated:
                val = model1.layers[i].get_weights()[0]
                # print(val.shape)
                
                vvv = np.zeros(shape=(val.shape[0], val.shape[1], sh, val.shape[3]), dtype=np.float32)
                if sh>1:
                    vvv[:, :, 0:sh-1, :] = val
                if copy:
                    vvv[:, :, sh-1, :] = val[:, :, 2, :]
                model.layers[i].set_weights([vvv])
                notUpdated = False
            else:
                try:
                    model.layers[i].set_weights(model1.layers[i].get_weights())
                except:
                    traceback.print_exc()

class KFoldCallback(keras.callbacks.Callback):

    def __init__(self, k:datasets.ImageKFoldedDataSet):
        super().__init__()
        self.data=k

    def on_epoch_end(self, epoch, logs=None):
        self.data.epoch()
        pass


class ReporterCallback(keras.callbacks.Callback):

    def __init__(self,reporter):
        super().__init__()
        self.reporter=reporter
        pass

    def on_batch_end(self, batch, logs=None):
        if self.reporter.isCanceled():
            self.model.stop_training = True
        super().on_batch_end(batch, logs)


class Stage:

    def __init__(self, dict, cfg: GenericTaskConfig):
        self.dict = dict
        self.cfg = cfg;
        self.negatives="all"
        if 'initial_weights' in dict:
            self.initial_weights=dict['initial_weights']
        else: self.initial_weights=None
        if 'negatives' in dict:
            self.negatives = dict['negatives']
        if 'validation_negatives' in dict:
            self.validation_negatives = dict['validation_negatives']
        else:
            self.validation_negatives=None
        self.epochs = dict["epochs"]
        if 'loss' in dict:
            self.loss = dict['loss']
        else:
            self.loss = None
        if 'lr' in dict:
            self.lr = dict['lr']
        else:
            self.lr = None

    def lr_find(self, kf: datasets.DefaultKFoldedDataSet, model: keras.Model, ec: ExecutionConfig, start_lr, end_lr, epochs):
        if 'unfreeze_encoder' in self.dict and self.dict['unfreeze_encoder']:
            self.unfreeze(model)

        if 'unfreeze_encoder' in self.dict and not self.dict['unfreeze_encoder']:
            self.freeze(model)

        if self.loss or self.lr:
            self.cfg.compile(model, self.cfg.createOptimizer(self.lr), self.loss)

        cb = [] + self.cfg.callbacks
        if self.initial_weights is not None:
            try:
                model.load_weights(self.initial_weights)
            except:
                z=model.layers[-1].name
                model.layers[-1].name="tmp"
                model.load_weights(self.initial_weights,by_name=True)
                model.layers[-1].name="z"
        ll=LRFinder(model)
        num_batches=kf.numBatches(ec.fold,self.negatives,ec.subsample)*epochs
        ll.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))
        K.set_value(model.optimizer.lr, start_lr)
        callback = LambdaCallback(on_batch_end=lambda batch, logs: ll.on_batch_end(batch, logs))
        cb.append(callback)
        kf.trainOnFold(ec.fold, model, cb,epochs, self.negatives, subsample=ec.subsample,validation_negatives=self.validation_negatives)
        return ll


    def _doTrain(self, kf, model, ec, cb, kepoch):
        return kf.trainOnFold(ec.fold, model, cb, self.epochs, self.negatives, subsample=ec.subsample, validation_negatives=self.validation_negatives, verbose=self.cfg.verbose, initial_epoch=kepoch)


    def _addLogger(self, model, ec, cb, kepoch):
        if self.cfg.resume:
            kepoch = maxEpoch(ec.metricsPath())
            if kepoch != -1:
                if os.path.exists(ec.weightsPath()):
                    model.load_weights(ec.weightsPath())
                cb.append(CSVLogger(ec.metricsPath(), append=True))
            else:
                cb.append(CSVLogger(ec.metricsPath()))
                kepoch = 0
        else:
            kepoch = 0
            cb.append(CSVLogger(ec.metricsPath()))
        return kepoch

    def execute(self, kf: datasets.DefaultKFoldedDataSet, model: keras.Model, ec: ExecutionConfig,callbacks=None):
        if 'unfreeze_encoder' in self.dict and self.dict['unfreeze_encoder']:
            self.unfreeze(model)

        if 'unfreeze_encoder' in self.dict and not self.dict['unfreeze_encoder']:
            self.freeze(model)
        if callbacks is None:
            if self.cfg.callbacks is not None:
                cb = [] + self.cfg.callbacks
            else:
                cb = []    
        else:
            cb=callbacks
        if self.cfg._reporter is not None:
            if self.cfg._reporter.isCanceled():
                return
            cb.append(ReporterCallback(self.cfg._reporter))
            pass
        prevInfo = None
        if self.cfg.resume:
            allBest = self.cfg.info()
            filtered = list(filter(lambda x: x.stage == ec.stage and x.fold == ec.fold, allBest))
            if len(filtered) > 0:
                prevInfo = filtered[0]
                self.lr = prevInfo.lr

        if self.loss or self.lr:
            self.cfg.compile(model, self.cfg.createOptimizer(self.lr), self.loss)
        if self.initial_weights is not None:
            try:
                    model.load_weights(self.initial_weights)
            except:
                    z=model.layers[-1].name
                    model.layers[-1].name="tmpName12312"
                    model.load_weights(self.initial_weights,by_name=True)
                    model.layers[-1].name=z
        if 'callbacks' in self.dict:
            cb = configloader.parse("callbacks", self.dict['callbacks'])
        if 'extra_callbacks' in self.dict:
            cb = cb + configloader.parse("callbacks", self.dict['extra_callbacks'])
        kepoch=-1
        if "logAll" in self.dict and self.dict["logAll"]:
            cb=cb+[AllLogger(ec.metricsPath()+"all.csv")]
        cb.append(KFoldCallback(kf))
        kepoch = self._addLogger(model, ec, cb, kepoch)
        md = self.cfg.primary_metric_mode

        if self.cfg.gpus==1:

            mcp = keras.callbacks.ModelCheckpoint(ec.weightsPath(), save_best_only=True,
                                                         monitor=self.cfg.primary_metric, mode=md, verbose=1)
            if prevInfo != None:
                mcp.best = prevInfo.best

            cb.append(mcp)

        self.add_visualization_callbacks(cb, ec, kf)
        if self.epochs-kepoch==0:
            return
        if self.cfg.gpus>1:
            omodel=model
            omodel.save_weights(ec.weightsPath()+".tmp",True)
            model=multi_gpu_model(model,self.cfg.gpus,True,True)
            lr=self.cfg.lr;
            if self.lr is not None:
                lr=self.lr
            loss=self.cfg.loss
            if self.loss is not None:
                loss=self.loss

            self.cfg.compile(model, self.cfg.createOptimizer(lr), loss)
            print("Restoring weights...")
            # weights are destroyed by some reason
            mda=None
            for q in model.layers:
                if isinstance(q,keras.Model):
                    mda=q
            bestWeightsLoaded = self.loadBestWeightsFromPrevStageIfExists(ec, mda)
            if not bestWeightsLoaded:
                mda.load_weights(ec.weightsPath()+".tmp",False)

            amcp = alt.AltModelCheckpoint(ec.weightsPath(), mda, save_best_only=True,
                                                monitor=self.cfg.primary_metric, mode=md, verbose=1)
            if prevInfo != None:
                amcp.best = prevInfo.best

            cb.append(amcp)
        else:
            self.loadBestWeightsFromPrevStageIfExists(ec, model)
        self._doTrain(kf, model, ec, cb, kepoch)

        print('saved')
        pass

    def loadBestWeightsFromPrevStageIfExists(self, ec, model):
        bestWeightsLoaded = False
        if ec.stage > 0:
            ec.stage = ec.stage - 1
            try:
                if os.path.exists(ec.weightsPath()):
                    print("Loading best weights from previous stage...")
                    model.load_weights(ec.weightsPath(), False)
                    bestWeightsLoaded = True
            except:
                pass
            ec.stage = ec.stage + 1
        return bestWeightsLoaded

    def unfreeze(self, model):
        pass

    def freeze(self, model):
        pass

    def add_visualization_callbacks(self, cb, ec, kf):
        pass
    
class MultiSplitStage(Stage):
    def _doTrain(self, kf, model:keras.Model, ec, cb, kepoch):
        bestIndex=-1
        for stage in range(0,10):
            
            coef=[0.1,0.2,0.3,0.5,1,1.5,2,3]
            if bestIndex!=-1:
                print("Loading from :",bestIndex)
                model.load_weights(ec.weightsPath()+"."+str(stage-1)+"."+str(bestIndex)+".weights",True)
            else:
                model.load_weights("D:/jigsaw/experiments/e9/weights/best.weights")
            model.save_weights(ec.weightsPath()+"."+"def."+str(stage)+".weights", True)    
            bestRes=100000
            bestIndex=-1
            list=[]
            for i in range(len(coef)):
                model.load_weights(ec.weightsPath()+"."+"def."+str(stage)+".weights", True)
                
                K.set_value(model.optimizer.lr, self.cfg.lr*coef[i])
                kf.trainOnIndexes(ec.fold, model, cb+[CSVLogger(ec.metricsPath()+"."+str(stage)+"."+str(i)+".csv"),AllLogger(ec.metricsPath()+"."+str(stage)+"."+str(i)+".all.csv")], "all",
                            stage,True)
                vvl=pd.read_csv(ec.metricsPath()+"."+str(stage)+"."+str(i)+".csv")
                res=vvl["val_loss"].values[0]
                print(res)
                if res<bestRes:
                    bestRes=res
                    bestIndex=i
                    model.save_weights(ec.weightsPath()+"."+str(stage)+"."+str(i)+".weights", True)
                list.append([self.cfg.lr*coef[i],res])
                print(list)    
                
            
            
            
    def _addLogger(self, model, ec, cb, kepoch):
        return 0        
            
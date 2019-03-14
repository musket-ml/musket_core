import os
import numpy as np
import yaml
import traceback
import csv
import keras
import tqdm
import pandas as pd

import musket_core.image_datasets
import musket_core.splits
import musket_core.tasks as tasks
from musket_core.image_datasets import WithBackgrounds,Backgrounds,CropAndSplit
from musket_core.utils import ensure
from keras.utils import multi_gpu_model
from musket_core.quasymodels import AnsembleModel,BatchCrop
import musket_core.datasets as datasets
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
import musket_core
from musket_core.clr_callback import CyclicLR
keras.callbacks.CyclicLR= CyclicLR
from musket_core import predictions
keras.utils.get_custom_objects()["macro_f1"]= musket_core.losses.macro_f1
keras.utils.get_custom_objects()["f1_loss"]= musket_core.losses.f1_loss
keras.utils.get_custom_objects()["dice"]= musket_core.losses.dice
keras.utils.get_custom_objects()["iou"]= musket_core.losses.iou_coef
keras.utils.get_custom_objects()["iot"]= musket_core.losses.iot_coef
keras.utils.get_custom_objects()["lovasz_loss"]= musket_core.losses.lovasz_loss
keras.utils.get_custom_objects()["iou_loss"]= musket_core.losses.iou_coef_loss
keras.utils.get_custom_objects()["dice_loss"]= musket_core.losses.dice_coef_loss
keras.utils.get_custom_objects()["jaccard_loss"]= musket_core.losses.jaccard_distance_loss
keras.utils.get_custom_objects()["focal_loss"]= musket_core.losses.focal_loss
keras.utils.get_custom_objects()["l2_loss"]= musket_core.losses.l2_loss

keras.utils.get_custom_objects().update({'matthews_correlation': musket_core.losses.matthews_correlation})
dataset_augmenters={

}
extra_train={}

class Rotate90(imgaug.augmenters.Affine):
    def __init__(self, enabled):
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

def threshold_search(y_true, y_proba,func):
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
        score = func(y_true.astype(np.float64), (y_proba > threshold))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    return ScoreAndTreshold(best_score,best_threshold)

def eval_metric(y_true,y_proba,func):
    if isinstance(func,str):
        func_=keras.metrics.get(func)
        def wrapped(x,y):

            v1= K.constant(x)
            v2 = K.constant(y)
            return K.eval(func_(v1,v2))
        func=wrapped
    return func(y_true,y_proba)

class GenericTaskConfig:

    def __init__(self,**atrs):
        self.batch = 8
        self.holdoutArr=None
        self.all = atrs
        self.groupFunc=None
        self.imports=[]
        self._dataset=None
        self.testTimeAugmentation=None
        self.stratified=False
        self.preprocessing=None
        self.verbose = 1
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
        self.gpus = 1
        self.lr = 0.001
        self.callbacks = []
        self.declarations={}
        self.optimizer = None
        self.loss = None
        self.testSplit = 0
        self.testSplitSeed = 123
        self.path = None
        self.metrics = []
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
        self.dropout = 0
        self.dataset_clazz = datasets.DefaultKFoldedDataSet
        for v in atrs:
            val = atrs[v]
            val = self._update_from_config(v, val)
            setattr(self, v, val)
        pass

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
        if ho is not None:
            train = list(set(range(1,len(ds))).difference(self.holdoutArr))

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

    def kfold(self, ds=None, indeces=None,batch=None)-> datasets.DefaultKFoldedDataSet:
        if ds is None:
            ds=self.get_dataset()
        if self.testSplit>0:
            if os.path.exists(self.path + ".holdout_split"):
                trI,hI = utils.load_yaml(self.path + ".holdout_split")
                train=datasets.SubDataSet(ds,trI)
                test = datasets.SubDataSet(ds,hI)
            else:
                train,test=self.doGetHoldout(ds)
                utils.save_yaml(self.path + ".holdout_split",(train.indexes,test.indexes))
            ds=train
            pass
        if batch is None:
            batch=self.batch
        if indeces is None: indeces=range(0,len(ds))
        transforms = [] + self.transforms
        ds = self.inject_task_specific_transforms(ds, transforms)
        split_loaded=False
        if os.path.exists(self.path+".folds_split"):
             folds=utils.load_yaml(self.path+".folds_split")
             split_loaded=True
             ds.folds=folds
        kf= self.dataset_clazz(ds, indeces, self.augmentation, transforms, batchSize=batch,rs=self.random_state,folds=self.folds_count,stratified=self.stratified,groupFunc=self.groupFunc)
        if not split_loaded:
            kf.save(self.path+".folds_split")
        if self.noTrain:
            kf.clear_train()
        if self.extra_train_data is not None:
            kf.addToTrain(extra_train[self.extra_train_data])
        if self.dataset_augmenter is not None:
            args = dict(self.dataset_augmenter)
            del args["name"]
            ag=dataset_augmenters[self.dataset_augmenter["name"]](**args)
            kf=ag(kf)
            pass
        return kf

    def predict_on_dataset(self, dataset, fold=0, stage=0, limit=-1, batch_size=32, ttflips=False):
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
        with tqdm.tqdm(total=len(dataset), unit="files", desc="prediiction from  " + str(dataset)) as pbar:
            for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips):
                b=v
                for i in range(len(b.data)):
                    res.append(b.results[i])
                pbar.update(batch_size)
        return np.array(res)

    def predictions(self,name,fold=None,stage=None):
        return musket_core.predictions.get_predictions(self,name,fold,stage)

    def find_treshold(self,ds,fold,func,stage=0):

        if isinstance(stage,list) or isinstance(stage,tuple):
            pa = []
            for i in stage:
                pa.append(self.predict_all_to_array(ds, fold, i))
            predicted = np.mean(np.array(pa),axis=0)
        else: predicted = self.predict_all_to_array(ds, fold, stage)
        vl = datasets.get_targets_as_array(ds)
        return threshold_search(vl, predicted,func)

    def find_optimal_treshold_by_validation(self,func,stages=None):
        tresh = []
        for fold in range(self.folds_count):
            arr = predictions.get_validation_prediction(self,fold,stages)
            val = self.validation( fold)
            targets = datasets.get_targets_as_array(val)
            tr = threshold_search(targets,arr , func)
            tresh.append(tr.treshold)
        tr = np.mean(np.array(tresh))
        return tr

    def find_optimal_treshold_by_validation2(self,func,stages=None,ds=None,):
        all=[]
        allTargets=[]
        for fold in range(self.folds_count):
            val = self.validation(ds, fold)
            preds=predictions.get_validation_prediction(self,fold,stages)
            targets=datasets.get_targets_as_array(val)
            all.append(preds)
            allTargets.append(targets)

        tr = threshold_search(np.concatenate(allTargets,axis=0),np.concatenate(all,axis=0),func)
        return tr.treshold

    def find_optimal_treshold_by_holdout(self,func,stages=None):
        hl=predictions.get_holdout_prediction(self,None,stages)
        return threshold_search(datasets.get_targets_as_array(self.holdout()), hl, func).treshold


    def createAndCompile(self, lr=None, loss=None)->keras.Model:
        return self.compile(self.createNet(), self.createOptimizer(lr=lr), loss=loss)

    def createNet(self):
        raise ValueError("Not implemented")

    def validation(self,ds=None,foldNum=0):
        if ds is None:
            ds = self.get_dataset()
        if isinstance(ds,int):
            foldNum=ds
            ds=self.get_dataset()
        ids=self.kfold(ds).indexes(foldNum,False)
        return datasets.SubDataSet(ds,ids)

    def train(self,ds,foldNum):
        ids=self.kfold(ds).indexes(foldNum,True)
        return datasets.SubDataSet(ds,ids)

    def createOptimizer(self, lr=None):
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
            return self.parse_dataset(dataSetName)


        if self._dataset is not None:
            return self._dataset
        if self.dataset is not None:
            self._dataset=self.parse_dataset()
            return self._dataset
        raise ValueError("Data set is not defined for this config")

    def fit(self, dataset_ = None, subsample=1.0, foldsToExecute=None, start_from_stage=0, drawingFunction = None):
        if dataset_ is None:
          dataset = self.parse_dataset()
        else: dataset=dataset_

        dataset = self._adapt_before_fit(dataset)
        self._dataset=dataset

        dn = self.directory()
        if os.path.exists(constructSummaryYamlPath(dn)):
            raise ValueError("Experiment is already finished!!!!")
        folds = self.kfold(dataset, None)
        for i in range(len(folds.folds)):
            if foldsToExecute:
                if not i in foldsToExecute:
                    continue
            model = self.createAndCompile()
            for s in range(0, len(self.stages)):
                if s<start_from_stage:
                    self.skip_stage(i, model, s, subsample)
                    continue
                st: Stage = self.stages[s]
                ec = ExecutionConfig(fold=i, stage=s, subsample=subsample, dr=self.directory(), drawingFunction=drawingFunction)
                st.execute(folds, model, ec)
        self.generateReports(foldsToExecute, subsample)



    def generateReports(self, foldsToExecute=None, subsample=1.0):
        dn = self.directory()
        with open(constructSummaryYamlPath(dn), "w") as f:
            initial = {"completed": True, "cfgName": os.path.basename(self.path), "subsample": subsample,
                       "folds": foldsToExecute}
            metrics = self.createSummary(foldsToExecute,subsample)
            for k in metrics:
                initial[k] = metrics[k]
            yaml.dump(
                initial,
                f)

    def createSummary(self,foldsToExecute, subsample):
        stagesStat=[]
        metric = self.primary_metric
        if "val_" in metric:
            metric=metric[4:]
        for stage in range(len(self.stages)):
            ms={}
            #tr=self.find_optimal_treshold_by_validation2(metric, [stage])
            #tr0 = self.find_optimal_treshold_by_validation(metric, [stage])
            for m in self.metrics:
               ms[m]=predictions.cross_validation_stat(self,m,[stage])

               if self.testSplit > 0:
                   ms[m + "_holdout"] = predictions.holdout_stat(self, m, [stage])
               if self.testSplit > 0:
                   ms[m + "_holdout"] = predictions.holdout_stat(self, m)
                   #ms[m + "_holdout_with_optimal_treshold"] = predictions.holdout_stat(self, m, [stage],tr)
                   #ms[m + "_holdout_with_optimal_treshold_mean"] = predictions.holdout_stat(self, m, [stage], tr0)
            stagesStat.append(ms)
        all={}
        tr = self.find_optimal_treshold_by_validation2(metric)
        #tr0 = self.find_optimal_treshold_by_validation(metric)
        for m in self.metrics:
            all[m] = predictions.cross_validation_stat(self, m)
            if self.testSplit > 0 or self.holdoutArr is not None:
                all[m + "_holdout"] = predictions.holdout_stat(self, m)
                all[m + "_holdout_with_optimal_treshold"] = predictions.holdout_stat(self, m, None, tr)
                #all[m + "_holdout_with_optimal_treshold_mean"] = predictions.holdout_stat(self, m, None, tr0)
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

    def parse_dataset(self):
        ds_config = self.pickup_ds_config()

        return musket_core.image_datasets.DS_Wrapper(self.dataset, ds_config, self.path)

    def pickup_ds_config(self):
        return self.datasets

    def clean(self, cleaned):
        cleaned.pop("datasets", None)
        cleaned.pop("dataset", None)
        cleaned.pop("run_tasks", None)
        cleaned.pop("import_tasks", None)

class TaskConfigInfo:

    def __init__(self,fold,stage,best,lr):
        self.fold = fold
        self.stage = stage
        self.best = best
        self.lr = lr


class GenericImageTaskConfig(GenericTaskConfig):

    def __init__(self,**atrs):
        super().__init__(**atrs)

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

    def predict_on_dataset(self, dataset, fold=0, stage=0, limit=-1, batch_size=None, ttflips=False):
        if batch_size is None:
            batch_size=self.inference_batch
        mdl = self.load_model(fold, stage)
        if self.crops is not None:
            mdl=BatchCrop(self.crops,mdl)
        ta = self.transformAugmentor()
        for original_batch in datasets.batch_generator(dataset, batch_size, limit):
            for batch in ta.augment_batches([original_batch]):
                res = self.predict_on_batch(mdl, ttflips, batch)
                self.update(batch,res)
                yield batch



    def _adapt_before_fit(self, dataset):
        if self.crops is not None:
            dataset = CropAndSplit(dataset, self.crops)
        return dataset

    def update(self,batch,res):
        pass

    def predict_on_batch(self, mdl, ttflips, batch):
        o1 = np.array(batch.images_aug)
        res = mdl.predict(o1)
        if ttflips == "Horizontal":
            another = imgaug.augmenters.Fliplr(1.0).augment_images(batch.images_aug)
            res1 = mdl.predict(np.array(another))
            if self.flipPred:
                res1 = imgaug.augmenters.Fliplr(1.0).augment_images(res1)
            res = (res + res1) / 2.0
        elif ttflips:
            res = self.predict_with_all_augs(mdl, ttflips, batch)
        return res
    
    def predict_with_all_augs(self, mdl, ttflips, batch):
        input_left = batch.images_aug
        input_right = imgaug.augmenters.Fliplr(1.0).augment_images(batch.images_aug)
        
        out_left = self.predict_with_all_rot_augs(mdl, ttflips,  input_left)
        out_right = self.predict_with_all_rot_augs(mdl, ttflips,  input_right)
        
        if self.flipPred:
            out_right = imgaug.augmenters.Fliplr(1.0).augment_images(out_right)
        
        return (out_left + out_right) / 2.0

    def predict_with_all_rot_augs(self, mdl, ttflips,  input):
        rot_90 = imgaug.augmenters.Affine(rotate=90.0)
        rot_180 = imgaug.augmenters.Affine(rotate=180.0)
        rot_270 = imgaug.augmenters.Affine(rotate=270.0)
        count = 2.0
        
        res_0 = mdl.predict(np.array(input))
        
        res_180 = self.predict_there_and_back(mdl, rot_180, rot_180, input)
        
        res_270 = 0;
        res_90 = 0
        
        if ttflips == "Horizontal_and_vertical":
            count = 4.0
            
            res_270 = self.predict_there_and_back(mdl, rot_270, rot_90, input)
            res_90 = self.predict_there_and_back(mdl, rot_90, rot_270, input)
        
        return (res_0 + res_90 + res_180 + res_270) / count
    
    def predict_there_and_back(self, mdl, there, back, input):
        augmented_input = there.augment_images(input)
        there_res = mdl.predict(np.array(augmented_input))
        if self.flipPred:
            return back.augment_images(there_res)
        return there_res

    def inject_task_specific_transforms(self, ds, transforms):
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
                    z.predictions = res;
                    pbar.update(batch_size)
                    yield z

    def transformAugmentor(self):
        transforms = [] + self.transforms
        transforms.append(imgaug.augmenters.Scale({"height": self.shape[0], "width": self.shape[1]}))
        return imgaug.augmenters.Sequential(transforms)


    def adaptNet(self, model, model1, copy=False):
        notUpdated = True
        for i in range(0, len(model1.layers)):
            if isinstance(model.layers[i], keras.layers.BatchNormalization) and notUpdated:
                uw = []
                for w in model1.layers[i].get_weights():
                    val = w
                    vvv = np.zeros(shape=(4), dtype=np.float32)
                    vvv[0:3] = val
                    vvv[3] = (val[0] + val[1] + val[2]) / 3
                    uw.append(vvv)
                model.layers[i].set_weights(uw)

            elif isinstance(model.layers[i], keras.layers.Conv2D) and notUpdated:
                val = model1.layers[i].get_weights()[0]
                # print(val.shape)
                vvv = np.zeros(shape=(val.shape[0], val.shape[1], 4, val.shape[3]), dtype=np.float32)
                vvv[:, :, 0:3, :] = val
                if copy:
                    vvv[:, :, 3, :] = val[:, :, 2, :]
                model.layers[i].set_weights([vvv])
                notUpdated = False
            else:
                try:
                    model.layers[i].set_weights(model1.layers[i].get_weights())
                except:
                    traceback.print_exc()

class KFoldCallback(keras.callbacks.Callback):

    def __init__(self, k:datasets.ImageKFoldedDataSet):
        self.data=k

    def on_epoch_end(self, epoch, logs=None):
        self.data.epoch()
        pass


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
            model.load_weights(self.initial_weights)
        ll=LRFinder(model)
        num_batches=kf.numBatches(ec.fold,self.negatives,ec.subsample)*epochs
        ll.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))
        K.set_value(model.optimizer.lr, start_lr)
        callback = LambdaCallback(on_batch_end=lambda batch, logs: ll.on_batch_end(batch, logs))
        cb.append(callback)
        kf.trainOnFold(ec.fold, model, cb,epochs, self.negatives, subsample=ec.subsample,validation_negatives=self.validation_negatives)
        return ll

    def execute(self, kf: datasets.DefaultKFoldedDataSet, model: keras.Model, ec: ExecutionConfig):
        if 'unfreeze_encoder' in self.dict and self.dict['unfreeze_encoder']:
            self.unfreeze(model)

        if 'unfreeze_encoder' in self.dict and not self.dict['unfreeze_encoder']:
            self.freeze(model)

        prevInfo = None
        if self.cfg.resume:
            allBest = self.cfg.info()
            filtered = list(filter(lambda x: x.stage == ec.stage and x.fold == ec.fold, allBest))
            if len(filtered) > 0:
                prevInfo = filtered[0]
                self.lr = prevInfo.lr

        if self.loss or self.lr:
            self.cfg.compile(model, self.cfg.createOptimizer(self.lr), self.loss)
        cb = [] + self.cfg.callbacks
        if self.initial_weights is not None:
            model.load_weights(self.initial_weights)
        if 'callbacks' in self.dict:
            cb = configloader.parse("callbacks", self.dict['callbacks'])
        if 'extra_callbacks' in self.dict:
            cb = cb + configloader.parse("callbacks", self.dict['extra_callbacks'])
        kepoch=-1
        cb.append(KFoldCallback(kf))
        if self.cfg.resume:
            kepoch=maxEpoch(ec.metricsPath())
            if kepoch!=-1:
                if os.path.exists(ec.weightsPath()):
                    model.load_weights(ec.weightsPath())
                cb.append(CSVLogger(ec.metricsPath(),append=True))
            else:
                cb.append(CSVLogger(ec.metricsPath()))
                kepoch=0
        else:
            kepoch=0
            cb.append(CSVLogger(ec.metricsPath()))
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
            bestWeightsLoaded = self.loadBestWeightsFromPrevStageIfExists(ec, model.layers[-2])
            if not bestWeightsLoaded:
                model.layers[-2].load_weights(ec.weightsPath()+".tmp",False)

            amcp = alt.AltModelCheckpoint(ec.weightsPath(), model.layers[-2], save_best_only=True,
                                                monitor=self.cfg.primary_metric, mode=md, verbose=1)
            if prevInfo != None:
                amcp.best = prevInfo.best

            cb.append(amcp)
        else:
            self.loadBestWeightsFromPrevStageIfExists(ec, model)
        kf.trainOnFold(ec.fold, model, cb, self.epochs, self.negatives, subsample=ec.subsample,validation_negatives=self.validation_negatives,verbose=self.cfg.verbose, initial_epoch=kepoch)

        print('saved')
        pass

    def loadBestWeightsFromPrevStageIfExists(self, ec, model):
        bestWeightsLoaded = False
        if ec.stage > 0:
            ec.stage = ec.stage - 1;
            try:
                if os.path.exists(ec.weightsPath()):
                    print("Loading best weights from previous stage...")
                    model.load_weights(ec.weightsPath(), False)
                    bestWeightsLoaded = True
            except:
                pass
            ec.stage = ec.stage + 1;
        return bestWeightsLoaded

    def unfreeze(self, model):
        pass

    def freeze(self, model):
        pass

    def add_visualization_callbacks(self, cb, ec, kf):
        pass
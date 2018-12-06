import os
import numpy as np
import yaml
import csv
import keras
import tqdm
import pandas as pd
from keras.utils import multi_gpu_model
from musket_core.quasymodels import AnsembleModel,BatchCrop
import musket_core.datasets as datasets
from musket_core import losses, configloader
import keras.optimizers as opt
from musket_core.lr_finder import LRFinder
from musket_core.logger import CSVLogger
import musket_core.multigpu_checkpoint as alt
from keras.callbacks import  LambdaCallback
import keras.backend as K
import imgaug
import musket_core
from musket_core.clr_callback import CyclicLR
keras.callbacks.CyclicLR= CyclicLR
keras.utils.get_custom_objects()["macro_f1"]= musket_core.losses.macro_f1
keras.utils.get_custom_objects()["dice"]= musket_core.losses.dice
keras.utils.get_custom_objects()["iou"]= musket_core.losses.iou_coef
keras.utils.get_custom_objects()["iot"]= musket_core.losses.iot_coef
keras.utils.get_custom_objects()["lovasz_loss"]= musket_core.losses.lovasz_loss
keras.utils.get_custom_objects()["iou_loss"]= musket_core.losses.iou_coef_loss
keras.utils.get_custom_objects()["dice_loss"]= musket_core.losses.dice_coef_loss
keras.utils.get_custom_objects()["jaccard_loss"]= musket_core.losses.jaccard_distance_loss
keras.utils.get_custom_objects()["focal_loss"]= musket_core.losses.focal_loss
dataset_augmenters={

}
extra_train={}

def ensure(p):
    try:
        os.makedirs(p);
    except:
        pass

def maxEpoch(file):
    if not os.path.exists(file):
        return -1;
    with open(file, 'r') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
         epoch=-1;
         num=0;
         for row in spamreader:
             if num>0:
                epoch=max(epoch,int(row[0]))
             num = num + 1;
         return epoch;


class ExecutionConfig:

    def __init__(self, fold=0, stage=0, subsample=1.0, dr: str = ""):
        self.subsample = subsample
        self.stage = stage
        self.fold = fold
        self.dirName = dr
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


class GenericConfig:

    def __init__(self,**atrs):
        self.batch = 8
        self.all = atrs
        self.copyWeights=False
        self.saveLast=False
        self.augmentation = []
        self.transforms = []
        self.architecture=None
        self.folds_count = 5
        self.encoder_weights=None
        self.random_state = 33
        self.stages = []
        self.gpus = 1
        self.lr=0.001
        self.callbacks = []
        self.path = None
        self.primary_metric = "val_binary_accuracy"
        self.primary_metric_mode = "auto"
        self.dataset_augmenter = None
        self.add_to_train = None
        self.extra_train_data = None
        self.bgr = None
        self.rate = 0.5
        self.dataset_clazz=None
        self.showDataExamples = False
        self.crops = None
        self.resume = False
        self.weights=None
        self.flipPred=True
        self.loss=None
        self.testSplit=0
        self.testSplitSeed=123
        self.optimizer=None
        self.shape=None
        self.metrics=[]
        for v in atrs:
            val = atrs[v]
            if v == 'augmentation' and val is not None:
                if "BackgroundReplacer" in val:
                    bgr=val["BackgroundReplacer"]
                    aug=None
                    erosion=0
                    if "erosion" in bgr:
                        erosion=bgr["erosion"]
                    if "augmenters" in bgr:
                        aug=bgr["augmenters"]
                        aug = configloader.parse("augmenters", aug)
                        aug=imgaug.augmenters.Sequential(aug)
                    self.bgr= datasets.Backgrounds(bgr["path"], erosion=erosion, augmenters=aug)
                    self.bgr.rate = bgr["rate"]
                    del val["BackgroundReplacer"]
                val = configloader.parse("augmenters", val)
            if v == 'transforms':
                val = configloader.parse("augmenters", val)
            if v == 'callbacks':
                cs=[]
                val = configloader.parse("callbacks", val)
                if val is not None:
                    val=val+cs
            if v == 'stages':
                val = [self.createStage(x) for x in val]
            setattr(self, v, val)

    def createStage(self,x):
        return None

    def load_model(self, fold: int = 0, stage: int = -1):
        if isinstance(fold,list):
            mdl=[];
            for i in fold:
                mdl.append(self.load_model(i,stage))
            return AnsembleModel(mdl)
        if stage == -1: stage = len(self.stages) - 1
        ec = ExecutionConfig(fold=fold, stage=stage, subsample=1.0, dr=os.path.dirname(self.path))
        model = self.createAndCompile()
        model.load_weights(ec.weightsPath())
        return model

    def predict_on_directory(self, path, fold=0, stage=0, limit=-1, batch_size=32, ttflips=False):
        return self.predict_on_dataset(datasets.DirectoryDataSet(path), fold, stage, limit, batch_size, ttflips)

    def predict_on_dataset(self, dataset, fold=0, stage=0, limit=-1, batch_size=32, ttflips=False):
        mdl = self.load_model(fold, stage)
        if self.crops is not None:
            mdl=BatchCrop(self.crops,mdl)
        ta = self.transformAugmentor()
        for v in datasets.batch_generator(dataset, batch_size, limit):
            for z in ta.augment_batches([v]):
                res = self.predict_on_batch(mdl, ttflips, z)
                self.update(z,res)
                yield z

    def fit(self, d, subsample=1.0, foldsToExecute=None, start_from_stage=0):
        if self.crops is not None:
            d= datasets.CropAndSplit(d, self.crops)
        dn = os.path.dirname(self.path)
        if os.path.exists(os.path.join(dn, "summary.yaml")):
            raise ValueError("Experiment is already finished!!!!")
        folds = self.kfold(d, range(0, len(d)))
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
                ec = ExecutionConfig(fold=i, stage=s, subsample=subsample, dr=os.path.dirname(self.path))
                st.execute(folds, model, ec)

        with open(os.path.join(dn, "summary.yaml"), "w") as f:
            yaml.dump(
                {"completed": True, "cfgName": os.path.basename(self.path), "subsample": subsample,
                 "folds": foldsToExecute},
                f)

    def update(self,z,res):
        pass

    def createOptimizer(self, lr=None):
        r = getattr(opt, self.optimizer)
        ds = create_with(["lr", "clipnorm", "clipvalue"], self.all)
        if lr:
            ds["lr"] = lr
        return r(**ds)

    def predict_on_batch(self, mdl, ttflips, z):
        o1 = np.array(z.images_aug)
        res = mdl.predict(o1)
        if ttflips == "Horizontal":
            another = imgaug.augmenters.Fliplr(1.0).augment_images(z.images_aug)
            res1 = mdl.predict(np.array(another))
            if self.flipPred:
                res1 = imgaug.augmenters.Fliplr(1.0).augment_images(res1)
            res = (res + res1) / 2.0
        elif ttflips:
            another = imgaug.augmenters.Fliplr(1.0).augment_images(z.images_aug)
            res1 = mdl.predict(np.array(another))
            if self.flipPred:
                res1 = imgaug.augmenters.Fliplr(1.0).augment_images(res1)

            another1 = imgaug.augmenters.Flipud(1.0).augment_images(z.images_aug)
            res2 = mdl.predict(np.array(another1))
            if self.flipPred:
                res2 = imgaug.augmenters.Flipud(1.0).augment_images(res2)

            seq = imgaug.augmenters.Sequential([imgaug.augmenters.Fliplr(1.0), imgaug.augmenters.Flipud(1.0)])
            another2 = seq.augment_images(z.images_aug)
            res3 = mdl.predict(np.array(another2))
            if self.flipPred:
                res3 = seq.augment_images(res3)
            res = (res + res1 + res2 + res3) / 4.0
        return res

    def compile(self, net: keras.Model, opt: keras.optimizers.Optimizer, loss:str=None)->keras.Model:
        if loss==None:
            loss=self.loss
        if "+" in loss:
            loss= losses.composite_loss(loss)
        if loss=='lovasz_loss' and isinstance(net.layers[-1],keras.layers.Activation):
            net=keras.Model(net.layers[0].input,net.layers[-1].input);
        if loss:
            net.compile(opt, loss, self.metrics)
        else:
            net.compile(opt, self.loss, self.metrics)
        return net

    def kfold(self, ds, indeces=None,batch=None)-> datasets.KFoldedDataSet:
        if self.testSplit>0:
            train,test=datasets.split(ds,self.testSplit,self.testSplitSeed)
            pass
        if batch is None:
            batch=self.batch
        if indeces is None: indeces=range(0,len(ds))
        transforms = [] + self.transforms
        transforms.append(imgaug.augmenters.Scale({"height": self.shape[0], "width": self.shape[1]}))
        if self.bgr is not None:
            ds= datasets.WithBackgrounds(ds, self.bgr)
        kf= self.dataset_clazz(ds, indeces, self.augmentation, transforms, batchSize=batch,rs=self.random_state,folds=self.folds_count)
        if self.extra_train_data is not None:
            kf.addToTrain(extra_train[self.extra_train_data])
        if self.dataset_augmenter is not None:
            args = dict(self.dataset_augmenter)
            del args["name"]
            ag=dataset_augmenters[self.dataset_augmenter["name"]](**args)
            kf=ag(kf)
            pass
        return kf

    def validation(self,ds,foldNum):
        ids=self.kfold(ds).indexes(foldNum,False)
        return datasets.SubDataSet(ds,ids)

    def createAndCompile(self, lr=None, loss=None)->keras.Model:
        return self.compile(self.createNet(), self.createOptimizer(lr=lr), loss=loss)

    def predict_on_directory_with_model(self, mdl, path, limit=-1, batch_size=32, ttflips=False):
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

    def lr_find(self, d, foldsToExecute=None,stage=0,subsample=1.0,start_lr=0.000001,end_lr=1.0,epochs=5):
        dn = os.path.dirname(self.path)
        if os.path.exists(os.path.join(dn, "summary.yaml")):
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
                ec = ExecutionConfig(fold=i, stage=s, subsample=subsample, dr=os.path.dirname(self.path))
                return st.lr_find(folds, model, ec,start_lr,end_lr,epochs)

    def skip_stage(self, i, model, s, subsample):
        st: Stage = self.stages[s]
        ec = ExecutionConfig(fold=i, stage=s, subsample=subsample, dr=os.path.dirname(self.path))
        if os.path.exists(ec.weightsPath()):
            model.load_weights(ec.weightsPath())
            if 'unfreeze_encoder' in st.dict and st.dict['unfreeze_encoder']:
                st.unfreeze(model)

    def adaptNet(self,model,model1,copy=False):
        notUpdated=True
        for i in range(0, len(model1.layers)):
            if isinstance(model.layers[i],keras.layers.Conv2D) and notUpdated:
                val = model1.layers[i].get_weights()[0]
                #print(val.shape)
                vvv = np.zeros(shape=(val.shape[0], val.shape[1], 4, val.shape[3]), dtype=np.float32)
                vvv[:, :, 0:3, :] = val
                if copy:
                    vvv[:, :, 3, :] = val[:, :, 2, :]
                model.layers[i].set_weights([vvv])
                notUpdated=False
            else:
                model.layers[i].set_weights(model1.layers[i].get_weights())

    def setAllowResume(self,resume):
        self.resume=resume

    def info(self,metric=None):
        if metric is None:
            metric=self.primary_metric
        ln=self.folds_count
        res=[]
        for i in range(ln):
            for s in range(0, len(self.stages)):

                st: Stage = self.stages[s]
                ec = ExecutionConfig(fold=i, stage=s, dr=os.path.dirname(self.path))
                if (os.path.exists(ec.metricsPath())):
                    try:
                        fr=pd.read_csv(ec.metricsPath())
                        res.append((i,s,fr[metric].max()))
                    except:
                        pass
        return res

class Stage:

    def __init__(self, dict, cfg: GenericConfig):
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

    def lr_find(self, kf: datasets.KFoldedDataSet, model: keras.Model, ec: ExecutionConfig, start_lr, end_lr, epochs):
        if 'unfreeze_encoder' in self.dict and self.dict['unfreeze_encoder']:
            self.unfreeze(model)
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

    def execute(self, kf: datasets.KFoldedDataSet, model: keras.Model, ec: ExecutionConfig):
        if 'unfreeze_encoder' in self.dict and self.dict['unfreeze_encoder']:
            self.unfreeze(model)
        if self.loss or self.lr:
            self.cfg.compile(model, self.cfg.createOptimizer(self.lr), self.loss)
        cb = [] + self.cfg.callbacks
        if self.initial_weights is not None:
            model.load_weights(self.initial_weights)
        if 'callbacks' in self.dict:
            cb = configloader.parse("callbacks", self.dict['callbacks'])
        if 'extra_callbacks' in self.dict:
            cb = configloader.parse("callbacks", self.dict['extra_callbacks'])
        kepoch=-1
        if self.cfg.resume:
            kepoch=maxEpoch(ec.metricsPath())
            if kepoch!=-1:
                self.epochs=self.epochs-kepoch
                if os.path.exists(ec.weightsPath()):
                    model.load_weights(ec.weightsPath())
                cb.append(CSVLogger(ec.metricsPath(),append=True,start=kepoch))
            else:
                cb.append(CSVLogger(ec.metricsPath()))
                kepoch=0
        else:
            kepoch=0
            cb.append(CSVLogger(ec.metricsPath()))
        md = self.cfg.primary_metric_mode
        if self.cfg.gpus>1:
            cb.append(
                alt.AltModelCheckpoint(ec.weightsPath(), model,save_best_only=True, monitor=self.cfg.primary_metric,
                                                mode=md, verbose=1))
        else:
            cb.append(
                keras.callbacks.ModelCheckpoint(ec.weightsPath(), save_best_only=True, monitor=self.cfg.primary_metric,
                                                mode=md, verbose=1))

        self.add_visualization_callbacks(cb, ec, kf)
        if self.epochs-kepoch==0:
            return
        if self.cfg.gpus>1:
            omodel=model
            model=multi_gpu_model(model,self.cfg.gpus,True,True)
            lr=self.cfg.lr;
            if self.lr is not None:
                lr=self.lr
            loss=self.cfg.loss
            if self.loss is not None:
                loss=self.loss

            self.cfg.compile(model, self.cfg.createOptimizer(lr), loss)
            #TODO This is weird
            if self.initial_weights is not None:
                model.layers[-2].load_weights(self.initial_weights)
            if os.path.exists(ec.weightsPath()):
                model.layers[-2].load_weights(ec.weightsPath())

        kf.trainOnFold(ec.fold, model, cb, self.epochs-kepoch, self.negatives, subsample=ec.subsample,validation_negatives=self.validation_negatives)
        pass

    def unfreeze(self, model):
        pass

    def add_visualization_callbacks(self, cb, ec, kf):
        pass
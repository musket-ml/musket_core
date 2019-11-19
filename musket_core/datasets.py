def warn(*args, **kwargs):
    pass
import warnings
import typing
old=warnings.warn
warnings.warn = warn
import sklearn.model_selection as ms
import imgaug
warnings.warn=old
import os
import keras
import imageio
import numpy as np
import traceback
import random
import tqdm
import pandas as pd
from musket_core import utils

AUGMENTER_QUEUE_LIMIT=10
USE_MULTIPROCESSING=False

class PredictionItem:
    def __init__(self, path, x, y, prediction = None):
        self.x=x
        self.y=y
        self.id=path
        self.prediction=prediction

    def original(self):
       return self

    def rootItem(self):
        return self
    
    def item_id(self):
        return self.id


class DataSet:

    def __init__(self):
        self.parent=None

    def __getitem__(self, item)->PredictionItem:
        raise ValueError("Not implemented")

    def get_train_item(self, item)->PredictionItem:
        return self[item]

    def __len__(self):
        raise ValueError("Not implemented")

    def __str__(self):
        if hasattr(self,"name"):
            return getattr(self,"name")
        return "dataset"

    def get_target(self,item):
        return self[item].y
    
    def get_inputs_count(self):
        if hasattr(self, "_inputs_count"):
            return self._inputs_count
        it=self[0]
        if isinstance(it.x, list) or isinstance(it.x, tuple):
            self._inputs_count=len(it.x)
        else:
            self._inputs_count=1
        return self._inputs_count;    
    
    def get_outputs_count(self):
        if hasattr(self, "_outputs_count"):
            return self._inputs_count
        it=self[0]
        if isinstance(it.y, list) or isinstance(it.y, tuple):
            self._outputs_count=len(it.x)
        else:
            self._outputs_count=1 
        return self._outputs_count;               
    
    def get_stratify_class(self,item):
        if hasattr(self, "parent"):
            if self.parent is not None:
                return self.parent.get_stratify_class(item)
        return self.get_target(item)
    
    def root(self):
        if not hasattr(self,"parent"):
            return self
        if self.parent is None:
            return self
        if hasattr(self.parent,"root"):
            return self.parent.root()
        return self.parent

    def meta(self, item):
        raise ValueError("Not implemented")
    
    def encode(self,item,encode_y=False,treshold=0.5):
        if isinstance(item, PredictionItem):
            return self._encode_item(item, encode_y, treshold)                
        if isinstance(item, DataSet):
            return self._encode_dataset(item, encode_y, treshold)

    def _encode_item(self,item:PredictionItem,encode_y=False,treshold=0.5):    
        raise NotImplementedError("This should be implemented in sub classes")
    
    def _encode_dataset(self,ds,encode_y=False,treshold=0.5):
        res=[]            
        for i in tqdm.tqdm(range(len(ds)),"Encoding dataset"):
            q=ds[i]
            rs=self.encode(q,encode_y,treshold)
            if isinstance(rs, list):
                res=res+rs
            else:    
                res.append(rs);
        return self._create_dataframe(res)
    
    def _create_dataframe(self,items):
        raise NotImplementedError("This should be implemented in sub classes") 

    def dump(self,path,treshold=0.5,encode_y=False):
        res=self.root().encode(self,treshold=treshold,encode_y=encode_y)
        res.to_csv(path,index=False) 

class WriteableDataSet(DataSet):

    def append(self,item):
        raise ValueError("Not implemented")

    def commit(self):
        raise ValueError("Not implemented")
    
    def blend(self,ds,w=0.5)->DataSet:
        if isinstance(ds, str):
            from musket_core import generic
            return self.blend(generic.parse(ds).predictions(self.parent.name))
        return self._inner_blend(ds, w)
    
    def dump(self,path,treshold=0.5,encode_y=False):
        res=self.root().encode(self,treshold=treshold,encode_y=encode_y)
        res.to_csv(path,index=False) 
        
    def encode(self, item, encode_y=False, treshold=0.5):
        res=self.root().encode(item,treshold=treshold,encode_y=encode_y)
        return res;            
        
    def _inner_blend(self,ds,w=0.5):
        raise NotImplementedError("Not supported")                    

def get_id(d:DataSet)->str:
    if hasattr(d,"id"):
        try:
            return d.id()
        except:
            pass
        return d.id
    if hasattr(d, "name"):
        return getattr(d, "name")
    if hasattr(d, "origName"):
        return getattr(d, "origName")
    return "<unknown>"

def get_stages(d:DataSet)->typing.List[str]:
    ds=d
    while isinstance(ds, SubDataSet):
            ds = ds.ds
    result=[]
    while ds is not None:
        _id= get_id(ds)
        result.append(_id)
        if hasattr(ds,"parent"):
            if hasattr(ds,"subStages"):
                brs=ds.subStages()
                for v in brs:
                    if v not in result:
                        result.append(v)
            p=getattr(ds,"parent")
            ds=p
        else: break
    return result
  
def get_preprocessors(d:DataSet)->typing.List[str]:
    ds=d
    while isinstance(ds, SubDataSet):
        ds = ds.ds
    result=[]
    while ds is not None:
        result.append(ds)
        if hasattr(ds,"parent"):
            if hasattr(ds,"subStages"):
                brs=ds.subStages()
                for v in brs:
                    if v not in result:
                        result.append(v)
            p=getattr(ds,"parent")
            ds=p
        else: break
    return result  


def get_stage(d:DataSet,n:str)->DataSet:
  ind=[]
  ds=d
  def createSub(ds,ind):
      for j in range(1,len(ind)+1):
          ds=SubDataSet(ds,ind[len(ind)-j])
      return ds
  while isinstance(ds, SubDataSet):
        ind.append(ds.indexes)
        ds = ds.ds

  while ds is not None:
      id=get_id(ds)
      if id==n:
           return createSub(ds,ind)
      if hasattr(ds,"get_stage"):
          v=ds.get_stage(n)
          if v is not None:
              return createSub(v, ind)
      if hasattr(ds,"parent"):
          p=getattr(ds,"parent")
          ds=p
      else: break
  pass

class DataSetLoader:
    def __init__(self,dataset,indeces,batchSize=16,isTrain=True):
        self.dataset = dataset
        self.batchSize = batchSize
        self.indeces = indeces
        self.isTrain=isTrain

    def generator(self):
        i = 0
        bx = []
        by = []
        ids= []
        while True:
            if i == len(self.indeces):
                i = 0
            try:
                id, x, y = self.proceed(i)
            except:
                traceback.print_exc()
                i = i + 1
                continue
            i = i + 1
            ids.append(id)
            bx.append(x)
            by.append(y)
            if len(bx) == self.batchSize:
                yield self.createBatch(bx, by, ids)
                bx = []
                by = []
                ids= []

    def createBatch(self, bx, by, ids):
        if isinstance(by[0],list):
            r=imgaug.augmentables.Batch(data=[ids,by], images=bx)
            return r
        if len(by[0].shape)>1:
            return imgaug.augmentables.Batch(data=ids, images=bx,
                                       segmentation_maps=[imgaug.SegmentationMapOnImage(x, shape=x.shape) for x
                                                          in by])
        else:
            r=imgaug.augmentables.Batch(data=[ids,by], images=bx)
            return r

    def load(self):
        i=0
        bx=[]
        by=[]
        ids = []
        while True:
            if (i==len(self.indeces)):
                i=0
            try:
                id, x, y = self.proceed(i)
            except:
                traceback.print_exc()
                i = i + 1
                continue
            ids.append(id)
            bx.append(x)
            by.append(y)
            i=i+1

            if len(bx)==self.batchSize:
                return self.createBatch(bx,by,ids)

    def proceed(self, i):
        id = ""
        if hasattr(self.dataset, "item"):
            item = self.dataset.item(self.indeces[i], self.isTrain)
        elif hasattr(self.dataset, "get_train_item") and self.isTrain:
            item = self.dataset.get_train_item(self.indeces[i])    
        else:
            item = self.dataset[self.indeces[i]]
        x, y = item.x, item.y
        if isinstance(item, PredictionItem):
            id = item.id
        return id, x, y


def drawBatch(batch,path):
    cells = []
    nc=2
    if not hasattr(batch, "segmentation_maps_aug") or batch.segmentation_maps_aug is None:
        batch.segmentation_maps_aug=batch.predicted_maps_aug
    if not hasattr(batch, "images_aug") or batch.images_aug is None:
        batch.images_aug=batch.images
        batch.segmentation_maps_aug=batch.predicted_maps_aug
    for i in range(0, len(batch.segmentation_maps_aug)):
        cells.append(batch.images_aug[i])
        if hasattr(batch,"predicted_maps_aug"):
            cells.append(batch.segmentation_maps[i].draw_on_image(batch.images_aug[i]))  # column 2
            nc=3
        cells.append(batch.segmentation_maps_aug[i].draw_on_image(batch.images_aug[i]))  # column 2
    # Convert cells to grid image and save.
    grid_image = imgaug.draw_grid(cells, cols=nc)
    imageio.imwrite(path, grid_image)

def draw_test_batch(batch,path):
    cells = []
    for i in range(0, len(batch.segmentation_maps_aug)):
        iPic = batch.images_aug[i][:, :, 0:3].astype(np.uint8)
        cells.append(iPic)
        cells.append(batch.segmentation_maps_aug[i].draw_on_image(iPic)[0])  # column 2
        cells.append(batch.heatmaps_aug[i].draw_on_image(iPic)[0])  # column 2
    # Convert cells to grid image and save.
    grid_image = imgaug.draw_grid(cells, cols=3)
    imageio.imwrite(path, grid_image)

class ConstrainedDirectory:
    def __init__(self,path,filters):
        self.path=path;
        self.filters=filters

    def __repr__(self):
        return self.path+" (with filter)"

class CompositeDataSet(DataSet):

    def __init__(self, components):
        super().__init__()
        self.components = components
        sum = 0;
        shifts = []
        for i in components:
            sum = sum + len(i)
            shifts.append(sum)
        self.shifts = shifts
        self.len = sum
        
    def get_stratify_class(self, item):
        i = item
        for j in range(len(self.shifts)):
            d = self.components[j]
            if item < self.shifts[j]:
                return d[i].get_stratify_class(item)
            else:
                i = item - self.shifts[j]
        raise ValueError("Should not happen")    
        
    def _encode_dataset(self,item:PredictionItem,encode_y=False,treshold=0.5):
        res=[]
        orD=None        
        for i in tqdm.tqdm(range(len(item)),"Encoding dataset"):
                q=item[i]
                ti=q;
                while not hasattr(ti, "originalDataSet") :ti=ti.original()
                orD=ti.originalDataSet
                
                rs=orD.encode(q,encode_y,treshold)
                if isinstance(rs, list):
                    res=res+rs
                else:    
                    res.append(rs);                
        res=orD._create_dataframe(res)                        
        return res 
                        

    def get_train_item(self, item)->PredictionItem:
        i = item
        for j in range(len(self.shifts)):
            d = self.components[j]
            if item < self.shifts[j]:
                if hasattr(d, "get_train_item"):
                    return d.get_train_item(i)
                res=d[i]
                res.originalDataSet=d
                return res 
            else:
                i = item - self.shifts[j]

        print("none")
        return None
    
    def get_target(self, item):
        if isinstance(item, slice):
            ifnone = lambda a, b: b if a is None else a
            rng = range(ifnone(item.start, 0), ifnone(item.stop, self.__len__()), ifnone(item.step, 1))
            result = [self.get_target(i) for i in rng]
            return result
        i = item
        for j in range(len(self.shifts)):
            provider = self.components[j]
            if item < self.shifts[j]:
                if hasattr(provider, "get_target"):
                    return provider.get_target(i)
                else:
                    return provider[i]
            else:
                i = item - self.shifts[j]

        return None


    def __getitem__(self, item):
        if isinstance(item, slice):
            ifnone = lambda a, b: b if a is None else a
            rng = range(ifnone(item.start, 0), ifnone(item.stop, self.__len__()), ifnone(item.step, 1))
            result = [self.__getitem__(i) for i in rng]
            return result
        i = item
        for j in range(len(self.shifts)):
            d = self.components[j]
            if item < self.shifts[j]:
                res=d[i]
                res.originalDataSet=d                
                return res 
            else:
                i = item - self.shifts[j]

        return None

    def isPositive(self, item):
        i = item
        for j in range(len(self.shifts)):
            d = self.components[j]
            if i < self.shifts[j]:
                return d.isPositive(i)
            else:
                i = i - self.shifts[j]
        return False

    def __len__(self):
        return self.len


def dataset_provider(origin=None,kind=None): 
    def inner(func): 
        func.dataset=True
        return func        
    return inner #this is the fun_obj mentioned in the above content 

class DirectoryDataSet(DataSet):

    def __init__(self,imgPath):
        super().__init__()
        if isinstance(imgPath,ConstrainedDirectory):
            self.imgPath=imgPath.path
            self.ids = imgPath.filters
        else:
            self.imgPath = imgPath;
            self.ids=os.listdir(imgPath)

        pass

    def __getitem__(self, item):
        return PredictionItem(self.ids[item], imageio.imread(os.path.join(self.imgPath,self.ids[item])),
                              None)

    def __len__(self):
        return len(self.ids)


def batch_generator(ds, batchSize, maxItems=-1):
        i = 0;
        bx = []
        ps = []
        im=len(ds)
        if maxItems!=-1:
            im=min(maxItems,im)
        for v in range(im):

            try:
                item = ds[i]
                x, y = item.x, item.id
            except:
                traceback.print_exc()
                i = i + 1
                continue
            i = i + 1
            bx.append(x)
            ps.append(y)
            if len(bx) == batchSize:
                yield imgaug.Batch(images=bx,data=ps)
                bx = []
                ps = []
        if len(bx)>0:
            yield imgaug.Batch(images=bx,data=ps)
        return
def generic_batch_generator(ds,batchSize,maxItems=-1):
    indexes=None
    if maxItems !=-1:
        indexes=list(range(min(maxItems,len(ds))))
    dg=GenericDataSetSequence(ds,batchSize,indexes,False)
    for i in range(len(dg)):
        zz=dg[i]
        if len(zz)==3:
            X,y,s=zz
        else:
            X,y=zz
        yield imgaug.Batch(images=X,data=y)
    return

class GenericDataSetSequence(keras.utils.Sequence):

    def __init__(self,ds,batch_size,indexes=None,infinite=True,isTrain=False):
        self.ds=ds
        self.batchSize=batch_size
        self._dim=None
        self.inifinite=infinite
        self.isTrain = isTrain
        if indexes is None:
            indexes =range(len(self.ds))
        self.indexes=indexes

    def __len__(self):
        return int(np.ceil(len(self.indexes) / float(self.batchSize)))

    def dim(self):
        if self._dim is not None:
            return self._dim
        v=self.ds[0]

        if isinstance(v.x, tuple) or isinstance(v.x, list):
            x_d=len(v.x)
        else: x_d=1

        if isinstance(v.y, tuple) or isinstance(v.y, list):
            y_d = len(v.y)
        else:
            y_d = 1
        self._dim=(x_d,y_d)
        return self._dim

    def __simple_batch(self, idx):
        l = len(self.indexes)
        X=[]
        y=[]
        hasSample=False
        if hasattr(self.ds, "root") and hasattr(self.ds.root(),"sample_weight"):
            hasSample=True
            S=[]
            
        for i in range(idx * self.batchSize,(idx + 1) * self.batchSize):
            if i>=l:
                i=i%l
                if not self.inifinite:
                    break
            r=self.ds.get_train_item(self.indexes[i]) if self.isTrain else self.ds[self.indexes[i]]
            X.append(r.x)
            y.append(r.y)
            if hasSample:
                S.append(self.ds.root().sample_weight(self.indexes[i]))
        if hasSample:        
            return np.array(X),np.array(y),np.array(S)
        return np.array(X),np.array(y)

    def on_epoch_end(self):
        random.shuffle(self.indexes)
        pass

    def __getitem__(self, idx):
        l=len(self.indexes)
        xd, yd=self.dim()
        if xd == 1 and yd==1:
            return self.__simple_batch(idx)

        batch_x = [[] for i in range(xd)]
        batch_y = [[] for i in range(yd)]
        hasSample=False
        if hasattr(self.ds.root(),"sample_weight"):
            hasSample=True
            S=[]
        for i in range(idx * self.batchSize,(idx + 1) * self.batchSize):
            if i>=l:
                i=i%l
                if not self.inifinite:
                    break
            r=self.ds.get_train_item(self.indexes[i]) if self.isTrain else self.ds[self.indexes[i]]
            for j in range(xd):
                r_x = r.x
                if not isinstance(r_x, list) and not isinstance(r_x, tuple):
                    r_x = [ r_x ]
                batch_x[j].append(r_x[j])
            for j in range(yd):
                r_y = r.y
                if not isinstance(r_y, list) and not isinstance(r_y, tuple):
                    r_y = [ r_y ]
                batch_y[j].append(r_y[j])
            if hasSample:
                S.append(self.ds.root().sample_weight(self.indexes[i]))    
        batch_x=[np.array(x) for x in batch_x]
        batch_y = np.array(batch_y[0]) if yd == 1 else [np.array(y) for y in batch_y]
        if hasSample:
#             sd=[]
#             
#             sd.append(np.array(S))
#             sd.append(np.ones(len(S)))        
            return batch_x,batch_y,np.array(S)
        
        return batch_x,batch_y

class SimplePNGMaskDataSet:
    def __init__(self, path, mask, detect_exts=False, in_ext="jpg", out_ext="png", generate=False,list=None):
        self.path = path
        self.mask = mask

        if list is None:
            ldir = os.listdir(path)

            if ".DS_Store" in ldir:
                ldir.remove(".DS_Store")

            self.ids = [x[0:x.index('.')] for x in ldir]
        else:
            self.ids=list

        self.exts = []

        if detect_exts:
            self.exts = [x[x.index('.') + 1:] for x in ldir]

        self.detect_exts = detect_exts

        self.in_ext = in_ext
        self.out_ext = out_ext

        self.generate = generate

        pass

    def __getitem__(self, item_):
        item = item_

        in_ext = self.in_ext
        out_ext = self.out_ext

        if self.detect_exts:
            in_ext = self.exts[item]
            # out_ext = self.exts[item]

        out_path = self.ids[item] + "." + out_ext

        # print("reading: " + os.path.join(self.mask, out_path))

        image = imageio.imread(os.path.join(self.path, self.ids[item] + "." + in_ext))

        out = []

        if not self.generate:
            out = imageio.imread(os.path.join(self.mask, out_path))
        else:
            out = np.zeros((image.shape[0], image.shape[1], 1))

        if len(out.shape) < 3:
            out = np.expand_dims(out, axis=2)

        out = out.astype(np.float32)

        out = np.sum(out, axis=2)

        out = np.expand_dims(out, axis=2)

        maxout = np.max(out)

        if maxout > 0:
            out = out / maxout

        if len(image.shape) < 3:
            image = np.expand_dims(image, 2)

            newImage = np.zeros((image.shape[0], image.shape[1], 3))

            newImage[:, :, 0] = image[:, :, 0]
            newImage[:, :, 1] = image[:, :, 0]
            newImage[:, :, 2] = image[:, :, 0]

            image = newImage

        elif image.shape[2] == 4:
            newImage = np.zeros((image.shape[0], image.shape[1], 3))

            newImage[:, :, 0] = image[:, :, 0]
            newImage[:, :, 1] = image[:, :, 1]
            newImage[:, :, 2] = image[:, :, 2]

            image = newImage
        
        image=image.astype(np.uint8)    
        return PredictionItem(self.ids[item] + str(), image, out>0.5)

    def isPositive(self, item):
        return True

    def __len__(self):
        return len(self.ids)

NB_WORKERS="auto"
NB_WORKERS_IN_LOADER=1
LOADER_SIZE=50
LOADER_THREADED=True


class DefaultKFoldedDataSet:
    def __init__(self,ds,indexes=None,aug=None,transforms=None,folds=5,rs=33,batchSize=16,stratified=True,groupFunc=None,validationSplit=0.2,maxEpochSize=None):
        self.ds=ds;
        if aug==None:
            aug=[]
        if transforms==None:
            transforms=[]
        self.aug=aug
        if indexes==None:
            indexes=range(len(ds))
        self.transforms=transforms
        self.batchSize=batchSize
        self.maxEpochSize = maxEpochSize
        self.positive={}
        if hasattr(ds,"folds"):
            self.folds=getattr(ds,"folds")
        else:
            
            if folds==1:
                
                if stratified:
                    classes=dataset_classes(ds,groupFunc)
                    r=ms.train_test_split(list(indexes),classes,shuffle=True,stratify=classes,random_state=rs,test_size=validationSplit)
                else: 
                    r=ms.train_test_split(list(indexes),shuffle=True,random_state=rs,test_size=validationSplit)
                self.folds=[r]
            else:    
                if stratified:
                    
                    self.kf = ms.StratifiedKFold(folds, shuffle=True, random_state=rs)
                    self.folds=[v for v in self.kf.split(indexes,dataset_classes(ds,groupFunc))]
                else:
                    self.kf = ms.KFold(folds, shuffle=True, random_state=rs)
                    self.folds = [v for v in self.kf.split(indexes)]

    def clear_train(self):
        nf = []
        for fold in self.folds:
            nf.append((fold[0][0:0],fold[1]))
        self.folds = nf

    def addToTrain(self,dataset):
        ma = len(self.ds)
        self.ds = CompositeDataSet([self.ds, dataset])
        nf = []
        for fold in self.folds:
            train = fold[0]
            rrr = np.concatenate([train, np.arange(ma, ma + len(dataset))])
            np.random.shuffle(rrr)
            nf.append((rrr, fold[1]))
        self.folds = nf

    def foldIterations(self,foldNum,isTrain=True):
        indexes = self.indexes(foldNum, isTrain)
        return len(indexes)//self.batchSize

    def indexes(self, foldNum, isTrain):
        fold = self.folds[foldNum]
        if isTrain:
            indexes = fold[0]
        else:
            indexes = fold[1]
        return indexes

    def epoch(self):
        pass

    def inner_isPositive(self,x):
        if x in self.positive:
            return self.positive[x]
        self.positive[x]=self.ds.isPositive(x)
        return self.positive[x]

    def generator(self,foldNum, isTrain=True,negatives="all",returnBatch=False):
        indexes = self.sampledIndexes(foldNum, isTrain, negatives)
        yield from self.generator_from_indexes(indexes,isTrain,returnBatch)

    def sampledIndexes(self, foldNum, isTrain, negatives):
        indexes = self.indexes(foldNum, isTrain)
        if negatives == 'none':
            indexes = [x for x in indexes if self.inner_isPositive(x)]
        if type(negatives)==int:
            sindexes = []
            nindexes = []
            for x in indexes:
                if self.inner_isPositive(x):
                    sindexes.append(x)
                else:
                    nindexes.append(x)
            random.seed(23232)
            random.shuffle(nindexes)
            nindexes = nindexes[ 0 : min(len(nindexes),round(len(sindexes)*negatives))]
            r=[]+sindexes+nindexes
            random.shuffle(r)
            return r;
        random.seed(23232)
        random.shuffle(indexes)
        return indexes

    def generator_from_indexes(self, indexes, isTrain=True, returnBatch=False):
        def _factory():
            return GenericDataSetSequence(self.ds,self.batchSize,indexes, isTrain = isTrain)
        return NullTerminatable(),NullTerminatable(),_factory

    def trainOnFold(self,fold:int,model:keras.Model,callbacks=[],numEpochs:int=100,negatives="all",
                    subsample=1.0,validation_negatives=None,verbose=1, initial_epoch=0):
        train_indexes = self.sampledIndexes(fold, True, negatives)
        if validation_negatives==None:
            validation_negatives=negatives
        test_indexes = self.sampledIndexes(fold, False, validation_negatives)

        tl,tg,train_g=self.generator_from_indexes(train_indexes)
        vl,vg,test_g = self.generator_from_indexes(test_indexes,isTrain=False)
        try:
            v_steps = len(test_indexes)//(round(subsample*self.batchSize))

            if v_steps < 1: v_steps = 1

            iterations = len(train_indexes) // (round(subsample * self.batchSize))
            if self.maxEpochSize is not None:
                iterations = min(iterations, self.maxEpochSize)
            model.fit_generator(train_g(), iterations,
                                epochs=numEpochs,
                                validation_data=test_g(),
                                callbacks=callbacks,
                                verbose=verbose,
                                validation_steps=v_steps,
                                initial_epoch=initial_epoch)
        finally:
            tl.terminate()
            tg.terminate()
            vl.terminate()
            vg.terminate()
            
    def trainOnIndexes(self,fold:int,model:keras.Model,callbacks,negatives,
                    indexes,doValidation=False):
        train_indexes = self.sampledIndexes(fold, True, negatives)
        vl=len(train_indexes)
        train_indexes=train_indexes[indexes*vl//10:(indexes+1)*vl//10]
        
        test_indexes = self.sampledIndexes(fold, False,negatives)

        tl,tg,train_g=self.generator_from_indexes(train_indexes) 
        vl,vg,test_g = self.generator_from_indexes(test_indexes,isTrain=False)
        try:
            v_steps = len(test_indexes)//self.batchSize

            if v_steps < 1: v_steps = 1

            iterations = len(train_indexes) // (self.batchSize)
            if self.maxEpochSize is not None:
                iterations = min(iterations, self.maxEpochSize)
            if doValidation:    
                model.fit_generator(train_g(), iterations,
                                    epochs=1,
                                    validation_data=test_g(),
                                    callbacks=callbacks,
                                    verbose=1,
                                    validation_steps=v_steps,
                                    initial_epoch=0)
            else:
                model.fit_generator(train_g(), iterations,
                                    epochs=1,
                                    callbacks=callbacks,
                                    verbose=1,                                    
                                    initial_epoch=0)    
        finally:
            tl.terminate()
            tg.terminate()
            vl.terminate()
            vg.terminate()        

    def numBatches(self,fold,negatives,subsample):
        train_indexes = self.sampledIndexes(fold, True, negatives)
        return len(train_indexes)//(round(subsample*self.batchSize))

    def save(self,path):
        utils.save_yaml(path,self.folds)


class ImageKFoldedDataSet(DefaultKFoldedDataSet):

    def epoch(self):
        for fold in self.folds:
            train = fold[0]
            np.random.shuffle(train)

    def load(self,foldNum, isTrain=True,negatives="all",ln=16):
        indexes = self.sampledIndexes(foldNum, isTrain, negatives)
        samples = DataSetLoader(self.ds, indexes, ln,isTrain=isTrain).load()
        for v in self.augmentor(isTrain).augment_batches([samples]):
            return v

    def _prepare_vals_from_batch(self, r):
        return np.array(r.images_aug), np.array([x.arr for x in r.segmentation_maps_aug])

    def generator_from_indexes(self, indexes,isTrain=True,returnBatch=False):
        aug = self.augmentor(isTrain)
        if len(aug)==0:
            return super().generator_from_indexes(indexes,isTrain,returnBatch)
        m = DataSetLoader(self.ds, indexes, self.batchSize,isTrain=isTrain).generator

        if USE_MULTIPROCESSING:
            l = imgaug.imgaug.BatchLoader(m,nb_workers=NB_WORKERS_IN_LOADER,threaded=LOADER_THREADED,queue_size=LOADER_SIZE)
            g = imgaug.imgaug.BackgroundAugmenter(l, augseq=aug,queue_size=AUGMENTER_QUEUE_LIMIT,nb_workers=NB_WORKERS)

            def r():
                num = 0
                while True:
                    r = g.get_batch()
                    x,y= self._prepare_vals_from_batch(r)
                    num=num+1
                    if returnBatch:
                        yield x,y,r
                    else: yield x,y

            return l,g,r
        else:
            def r():
                num = 0
                while True:
                    for batch in m():
                        r = list(aug.augment_batches([batch], background=False))[0]
                        x,y= self._prepare_vals_from_batch(r)
                        #Think about normalization
                        #x=(x/255.0-(0.485, 0.456, 0.406))
                        num=num+1
                        if returnBatch:
                            yield x,y,r
                        else: yield x,y

            return NullTerminatable(),NullTerminatable(),r

    def augmentor(self, isTrain)->imgaug.augmenters.Augmenter:
        allAug = []
        if isTrain:
            allAug = allAug + self.aug
        allAug = allAug + self.transforms
        aug = imgaug.augmenters.Sequential(allAug)
        return aug


class KFoldedDataSet4ImageClassification(ImageKFoldedDataSet):

    def _prepare_vals_from_batch(self, r):
        if isinstance(r.data[1][0],list):
            arra=[[] for x in range(len(r.data[1][0]))]
            for x in range(len(r.data[1])):
                rs=r.data[1][x]
                for j in range(len(arra)):
                    arra[j].append(rs[j])
            arra=[np.array(a) for a in arra]
            return np.array(r.images_aug), arra
        else:
            return np.array(r.images_aug), np.array([x for x in r.data[1]])


class NullTerminatable:

    def terminate(self):
        pass

class SubDataSet(DataSet):
    def __init__(self,orig,indexes):
        super().__init__()
        if isinstance(orig,int) or orig is None or isinstance(orig,list):
            raise ValueError("Dataset is expected")
        self.ds=orig
        self.parent=orig
        if hasattr(orig, "_preprocessed"):
            self._preprocessed=True
        self.indexes=indexes

    def isPositive(self, item):
        return self.ds.isPositive(self.indexes[item])


    def __getitem__(self, item):
        if isinstance(item, slice):
            result = [self.ds[i] for i in self.indexes[item]]
            return result
        else:
            return self.ds[self.indexes[item]]

    def get_train_item(self,item:int):
        if hasattr(self.ds, "get_train_item"):
            return self.ds.get_train_item(self.indexes[item])
        else:
            return self.ds[self.indexes[item]]
    
    def root(self):
        if hasattr(self.ds,"root"):
            return self.ds.root()
        return self.ds

    def __len__(self):
        return len(self.indexes)

    def get_target(self, item):
        if hasattr(self.parent,"get_target"):
            return self.parent.get_target(self.indexes[item])
        return self.parent[self.indexes[item]].y

def get_class_that_defined_method(method):
    method_name = method.__name__
    if method.__self__:    
        classes = [method.__self__.__class__]
    else:
        #unbound method
        classes = [method.im_class]
    while classes:
        c = classes.pop()
        if method_name in c.__dict__:
            return c
        else:
            classes = list(c.__bases__) + classes
    return None
def dataset_classes(ds, groupFunc):
    preds=[]
    ds=ds.root()
    if hasattr(ds,"get_stratify_class"):
        d=get_class_that_defined_method(getattr(ds, "get_stratify_class"))
        if (d!=DataSet):
            for i in tqdm.tqdm(range(len(ds)),"reading stratification classes "+str(ds)):
                preds.append(ds.get_stratify_class(i))
            return np.array(preds)    
    if groupFunc != None:
        data_classes = groupFunc(ds)
    else:
        data_classes = get_targets_as_array(ds);
        data_classes = data_classes.mean(axis=1) > 0
    return data_classes


def get_targets_as_array(d):
    preds=[]
    if hasattr(d,"get_target"):
        for i in tqdm.tqdm(range(len(d)),"reading dataset targets "+str(d)):
            preds.append(d.get_target(i))
    else:
        for i in tqdm.tqdm(range(len(d)),"reading dataset targets "+str(d)):
            preds.append(d[i].y)
    return np.array(preds,dtype=np.float32)



def inherit_dataset_params(ds_from,ds_to):
    if hasattr(ds_from, "folds"):
        ds_to.folds = getattr(ds_from, "folds")
    else:
        if hasattr(ds_to,"folds"):
            del ds_to.folds;    
    if hasattr(ds_from, "holdoutArr"):
        ds_to.holdoutArr = getattr(ds_from, "holdoutArr")
    if hasattr(ds_from, "contributions"):
        ds_to.contributions = getattr(ds_from, "contributions")    
    if hasattr(ds_from, "_preprocessed"):
        ds_to._preprocessed = getattr(ds_from, "_preprocessed")    
    if hasattr(ds_from, "contribution"):
        ds_to.contribution = getattr(ds_from, "contribution")    

class MergedDataSet:
    def __init__(self,components:[DataSet],mergeFunc=None):
        self.components = components
        self.mergeFunc = mergeFunc

    def isPositive(self, item):
        return self.ds.isPositive(self.indexes[item])

    def __getitem__(self, item):
        componentItems = [x[item] for x in self.components]
        isSlice = isinstance(item, slice)
        lst = list(zip(componentItems)) if isSlice else [ componentItems ]

        merged = [ self.mergeFunc(x) for x in lst ]
        result = merged if isSlice else merged[0]
        return result

    def __len__(self):
        return len(self.components[0])


def mergeFunc(items: [PredictionItem]) -> PredictionItem:
    preds = np.array([x.prediction for x in items])
    meanPred = np.mean(preds)
    i = items[0]
    result = PredictionItem(i.id, i.x, i.y, meanPred)
    return result

class MeanDataSet(MergedDataSet):
    def __init__(self,components:[DataSet]):
        super().__init__(components, mergeFunc)


class BufferedWriteableDS(WriteableDataSet):

    def __init__(self,orig,name,dsPath,predictions=None,pickle=False):
        super().__init__()
        if predictions is None:
            predictions = []
        self.parent = orig
        self.name=name
        self.pickle=pickle
        self.predictions=predictions
        self.dsPath=dsPath

    def append(self,item):
        self.predictions.append(item)

    def commit(self):
        if self.dsPath is not None:
            if self.pickle:
                utils.save(self.dsPath, self.predictions)
            else:
                np.save(self.dsPath,self.predictions)
        self.predictions=np.array(self.predictions)

    def __len__(self):
        return len(self.parent)
    
    def _inner_blend(self,ds,w=0.5):
        pr=self.predictions*w+ds.predictions*(1-w)
        return BufferedWriteableDS(self.parent,self.name,None,pr)

    def __getitem__(self, item):
        it = self.parent[item]
        if self.predictions is not None:
            if isinstance(item, slice):
                indSlice = list(range(len(self)))[item]
                for i in range(len(it)):
                    it[i].prediction = self.predictions[indSlice[i]]
            else:
                it.prediction = self.predictions[item]
        return it


class DirectWriteableDS(WriteableDataSet):

    def __init__(self,orig,name,dsPath, count = 0):
        super().__init__()
        self.parent = orig
        self.name=name
        self.dsPath=dsPath
        self.count = count

    def append(self,item):
        ip = self.itemPath(self.count)
        self.saveItem(ip,item)
        self.count += 1

    def commit(self):
        pass

    def __len__(self):
        return len(self.parent)
    
    def _inner_blend(self,ds,w=0.5):
        return WeightedBlend([self,ds],[w,1-w])

    def __getitem__(self, item):
        it = self.parent[item]
        if isinstance(item, slice):
            indSlice = list(range(len(self)))[item]
            for i in range(len(it)):
                ip = self.itemPath(indSlice[i])
                if os.path.exists(ip):
                    prediction = self.loadItem(ip)
                    it[i].prediction = prediction
        else:
            ip = self.itemPath(item)
            if os.path.exists(ip):
                prediction = self.loadItem(ip)
                it.prediction = prediction
        return it

    def itemPath(self, item:int)->str:
        return f"{self.dsPath}/{item}.npy"

    def saveItem(self, path:str, item):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.mkdir(dir)
        np.save(path, item)

    def loadItem(self, path:str):
        return np.load(path)
    
class CompressibleWriteableDS(WriteableDataSet):

    def __init__(self,orig,name,dsPath, count = 0,asUints=True,scale=255):
        super().__init__()
        self.parent = orig
        self.name=name
        self.dsPath=dsPath
        self.count = count
        self.asUints=asUints
        self.scale=scale

    def append(self,item):
        ip = self.itemPath(self.count)
        self.saveItem(ip,item)
        self.count += 1

    def commit(self):
        pass
    
    def _inner_blend(self,ds,w=0.5):
        return WeightedBlend([self,ds],[w,1-w])

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, item):
        it = self.parent[item]
        if isinstance(item, slice):
            indSlice = list(range(len(self)))[item]
            for i in range(len(it)):
                ip = self.itemPath(indSlice[i])
                if os.path.exists(ip):
                    prediction = self.loadItem(ip)
                    it[i].prediction = prediction
        else:
            ip = self.itemPath(item)
            if os.path.exists(ip):
                prediction = self.loadItem(ip)
                it.prediction = prediction
        return it

    def itemPath(self, item:int)->str:
        return f"{self.dsPath}/{item}.npy.npz"

    def saveItem(self, path:str, item):
        dire = os.path.dirname(path)
        if item is None:
            raise ValueError("Should never happen")
        if self.asUints:
            if self.scale<=255:
                item=(item*self.scale).astype(np.uint8)
            else:
                item=(item*self.scale).astype(np.uint16)    
        if not os.path.exists(dire):
            os.mkdir(dire)
        np.savez_compressed(path, item)

    def loadItem(self, path:str):
        if self.asUints:
            x=np.load(path)["arr_0.npy"].astype(np.float32)/self.scale
        else:
            x=np.load(path)["arr_0.npy"]      
        return x; 
    
class PredictionBlend(DataSet):
    
    def __init__(self,predictions,enableCache=False):
        self.predictions=predictions        
        self.cache={}    
        self.enableCache=enableCache
    
    def __len__(self):
        return len(self.predictions[0])
    
    
    def __getitem__(self, item)->PredictionItem:        
        if item in self.cache:
            return self.cache[item]        
        z=self.predictions[0][item]
        
        if z.prediction is None:
            return z
        pr=np.mean([v[item].prediction for v in self.predictions if v[item].prediction is not None],axis=0)
        r= PredictionItem(z.id,z.x,z.y,pr)
        if self.enableCache:
            self.cache[item]=r
        return r

class WeightedBlend(DataSet):
    
    def __init__(self,path,nm,weights=None):
        self.path=path
        self.nm=nm
        if weights is None:
            weights=[]
        if isinstance(path,list):
            v=path
            self.predictions=[]
            for path in v:
                
                
                self.predictions.append(path)
                if len(weights)<len(self.predictions):
                    weights.append(1)
        sw=sum(weights)
        self.weights=[x/sw for x in weights]
        self.cache={}    
    
    def __len__(self):
        return len(self.predictions[0])
    
    def blend(self,ds,w=0.5)->DataSet:
        if isinstance(ds, str):
            from musket_core import generic
            return self.blend(generic.parse(ds).predictions(self.parent.name))
        return self._inner_blend(ds, w)
    
    def _inner_blend(self,ds,w=0.5):
        return WeightedBlend([self,ds],[w,1-w])
    
    def dump(self,path,treshold=0.5,encode_y=False):
        res=self.predictions[0].encode(self,treshold=treshold,encode_y=encode_y)
        res.to_csv(path,index=False)
    
    def __getitem__(self, item)->PredictionItem:
        
        if item in self.cache:
            return self.cache[item]
        
        z=self.predictions[0][item]
        
        if z.prediction is None:
            return z
        
        prs=[]
        for i in range(len(self.predictions)):
            v=self.predictions[i]
            w=self.weights[i]
            if v[item].prediction is not None:
                prs.append(v[item].prediction*w)  
        pr=np.sum(prs,axis=0)
        #np.savez(self.nm+"/"+str(item),pr)
        r= PredictionItem(z.id,z.x,z.y,pr)
        #self.cache[item]=r
        return r     

class TransformPrediction(DataSet):
    
    def __init__(self,parent,func):
        self.parent=parent        
        self.func=func
    
    def __len__(self):
        return len(self.parent)
    
    
    def __getitem__(self, item)->PredictionItem:        
        z=self.parent[item]
        r= PredictionItem(z.id,z.x,z.y,self.func(z.prediction))
        return r
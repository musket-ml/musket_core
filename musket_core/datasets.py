def warn(*args, **kwargs):
    pass
import warnings
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
from musket_core import utils

AUGMENTER_QUEUE_LIMIT=10
USE_MULTIPROCESSING=False


class PredictionItem:
    def __init__(self, path, x, y):
        self.x=x
        self.y=y
        self.id=path

    def original(self):
       return self

    def rootItem(self):
        return self


class DataSet:
    def __getitem__(self, item)->PredictionItem:
        raise ValueError("Not implemented")

    def __len__(self):
        raise ValueError("Not implemented")

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
                #traceback.print_exc()
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
        if len(by[0].shape)>1:
            return imgaug.imgaug.Batch(data=ids, images=bx,
                                       segmentation_maps=[imgaug.SegmentationMapOnImage(x, shape=x.shape) for x
                                                          in by])
        else:
            r=imgaug.imgaug.Batch(data=[ids,by], images=bx)
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
        cells.append(batch.segmentation_maps_aug[i].draw_on_image(iPic))  # column 2
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

class CompositeDataSet:

    def __init__(self, components):
        self.components = components
        sum = 0;
        shifts = []
        for i in components:
            sum = sum + len(i)
            shifts.append(sum)
        self.shifts = shifts
        self.len = sum

    def item(self, item, isTrain):
        i = item
        for j in range(len(self.shifts)):
            d = self.components[j]
            if item < self.shifts[j]:
                if hasattr(d, "item"):
                    return d.item(i, isTrain)
                return d[i]
            else:
                i = item - self.shifts[j]

        print("none")
        return None

    def __getitem__(self, item):
        i = item
        for j in range(len(self.shifts)):
            d = self.components[j]
            if item < self.shifts[j]:
                return d[i]
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


class DirectoryDataSet:

    def __init__(self,imgPath):

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


class GenericDataSetSequence(keras.utils.Sequence):

    def __init__(self,ds,batch_size,indexes=None):
        self.ds=ds
        self.batchSize=batch_size
        self._dim=None
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
        for i in range(idx * self.batchSize,(idx + 1) * self.batchSize):
            if i>=l:
                i=i%l
            r=self.ds[self.indexes[i]]
            X.append(r.x)
            y.append(r.y)
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

        for i in range(idx * self.batchSize,(idx + 1) * self.batchSize):
            if i>=l:
                i=i%l
            r=self.ds[self.indexes[i]]
            for j in range(xd):
                r_x = r.x
                if not isinstance(r_x, list):
                    r_x = [ r_x ]
                batch_x[j].append(r_x[j])
            for j in range(yd):
                r_y = r.y
                if not isinstance(r_y, list):
                    r_y = [ r_y ]
                batch_y[j].append(r_y[j])
        batch_x=[np.array(x) for x in batch_x]
        batch_y = np.array(batch_y[0]) if yd == 1 else [np.array(y) for y in batch_y]
        return batch_x,batch_y

class SimplePNGMaskDataSet:
    def __init__(self, path, mask, detect_exts=False, in_ext="jpg", out_ext="png", generate=False):
        self.path = path;
        self.mask = mask;

        ldir = os.listdir(path)

        if ".DS_Store" in ldir:
            ldir.remove(".DS_Store")

        self.ids = [x[0:x.index('.')] for x in ldir]

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

        return PredictionItem(self.ids[item] + str(), image, out)

    def isPositive(self, item):
        return True

    def __len__(self):
        return len(self.ids)

NB_WORKERS="auto"
NB_WORKERS_IN_LOADER=1
LOADER_SIZE=50
LOADER_THREADED=True


class DefaultKFoldedDataSet:
    def __init__(self,ds,indexes=None,aug=None,transforms=None,folds=5,rs=33,batchSize=16,stratified=True,groupFunc=None):
        self.ds=ds;
        if aug==None:
            aug=[]
        if transforms==None:
            transforms=[]
        self.aug=aug;
        if indexes==None:
            indexes=range(len(ds))
        self.transforms=transforms
        self.batchSize=batchSize
        self.positive={}

        if hasattr(ds,"folds"):
            self.folds=getattr(ds,"folds");
        else:
            if stratified:
                self.kf = ms.StratifiedKFold(folds, shuffle=True, random_state=rs)
                self.folds=[v for v in self.kf.split(indexes,dataset_classes(ds,groupFunc))]
            else:
                self.kf = ms.KFold(folds, shuffle=True, random_state=rs);
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
        return indexes

    def generator_from_indexes(self, indexes, isTrain=True, returnBatch=False):
        def _factory():
            return GenericDataSetSequence(self.ds,self.batchSize,indexes)
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
            
            model.fit_generator(train_g(), len(train_indexes)//(round(subsample*self.batchSize)),
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
        m = DataSetLoader(self.ds, indexes, self.batchSize,isTrain=isTrain).generator
        aug = self.augmentor(isTrain)
        if USE_MULTIPROCESSING:
            l = imgaug.imgaug.BatchLoader(m,nb_workers=NB_WORKERS_IN_LOADER,threaded=LOADER_THREADED,queue_size=LOADER_SIZE)
            g = imgaug.imgaug.BackgroundAugmenter(l, augseq=aug,queue_size=AUGMENTER_QUEUE_LIMIT,nb_workers=NB_WORKERS)

            def r():
                num = 0;
                while True:
                    r = g.get_batch();
                    x,y= self._prepare_vals_from_batch(r)
                    num=num+1
                    if returnBatch:
                        yield x,y,r
                    else: yield x,y

            return l,g,r
        else:
            def r():
                num = 0;
                while True:
                    for batch in m():
                        r = list(aug.augment_batches([batch], background=False))[0]
                        x,y= self._prepare_vals_from_batch(r)
                        num=num+1
                        if returnBatch:
                            yield x,y,r
                        else: yield x,y

            return NullTerminatable(),NullTerminatable(),r

    def augmentor(self, isTrain)->imgaug.augmenters.Augmenter:
        allAug = [];
        if isTrain:
            allAug = allAug + self.aug
        allAug = allAug + self.transforms
        aug = imgaug.augmenters.Sequential(allAug);
        return aug


class KFoldedDataSet4ImageClassification(ImageKFoldedDataSet):

    def _prepare_vals_from_batch(self, r):
        return np.array(r.images_aug), np.array([x for x in r.data[1]])


class NullTerminatable:

    def terminate(self):
        pass

class SubDataSet:
    def __init__(self,orig,indexes):
        if isinstance(orig,int) or orig is None or isinstance(orig,list):
            raise ValueError("Dataset is expected")
        self.ds=orig
        self.indexes=indexes

    def isPositive(self, item):
        return self.ds.isPositive(self.indexes[item])

    def __getitem__(self, item):
        return self.ds[self.indexes[item]]

    def __len__(self):
        return len(self.indexes)


def dataset_classes(ds, groupFunc):
    if groupFunc != None:
        data_classes = groupFunc(ds)
    else:
        data_classes = np.array([ds[i].y for i in range(len(ds))]);
        data_classes = data_classes.mean(axis=1) > 0
    return data_classes


def get_targets_as_array(d):
    preds=[]
    for i in range(len(d)):
        preds.append(d[i].y)
    return np.array(preds)
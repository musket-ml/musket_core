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
import cv2 as cv

AUGMENTER_QUEUE_LIMIT=10
USE_MULTIPROCESSING=False

class PredictionItem:
    def __init__(self, path, x, y):
        self.x=x
        self.y=y
        self.id=path;

class DataSetLoader:
    def __init__(self,dataset,indeces,batchSize=16,isTrain=True):
        self.dataset = dataset
        self.batchSize = batchSize
        self.indeces = indeces
        self.isTrain=isTrain

    def generator(self):
        i = 0;
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
        i=0;
        bx=[]
        by=[]
        ids = []
        while True:
            if (i==len(self.indeces)):
                i=0;
            try:
                id, x, y = self.proceed(i)
            except:
                traceback.print_exc()
                i = i + 1;
                continue
            ids.append(id)
            bx.append(x)
            by.append(y)
            i=i+1;

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
        cells.append(batch.images_aug[i][:,:,0:3])
        cells.append(batch.segmentation_maps_aug[i].draw_on_image(batch.images_aug[i][:,:,0:3]))  # column 2
        cells.append(batch.heatmaps_aug[i].draw_on_image(batch.images_aug[i][:,:,0:3])[0])  # column 2
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


class Backgrounds:

    def __init__(self,path,erosion=0,augmenters:imgaug.augmenters.Augmenter=None):
        self.path=path;
        self.rate=0.5
        self.augs=augmenters
        self.erosion=erosion
        self.options=[os.path.join(path,x) for x in os.listdir(self.path)]

    def next(self,i,i2):
        fl=random.choice(self.options)
        im=imageio.imread(fl)
        r=cv.resize(im,(i.shape[1],i.shape[0]))
        if isinstance(self.erosion,list):
            er=random.randint(self.erosion[0],self.erosion[1])
            kernel = np.ones((er, er), np.uint8)
            i2 = cv.erode(i2, kernel)
        elif self.erosion>0:
            kernel = np.ones((self.erosion, self.erosion), np.uint8)
            i2=cv.erode(i2,kernel)
        i2=i2!=0
        i2=np.squeeze(i2)
        if i.shape[2]!=3:
           zr=np.copy(i)
           zr[:,:,0:3]=r
           zr[i2] = i[i2]
           return zr
        else:
            r[i2] = i[i2]
        return r;

    def augment_item(self,i):
        if self.augs!=None:

            b=imgaug.Batch(images=[i.x],
                                segmentation_maps=[imgaug.SegmentationMapOnImage(i.y, shape=i.y.shape)])
            for v in self.augs.augment_batches([b]):
                bsa:imgaug.Batch=v
                break
            xa=bsa.images_aug[0]

            xa=cv.resize(xa,(i.x.shape[1],i.x.shape[0]))
            ya=bsa.segmentation_maps_aug[0].arr
            ya = cv.resize(ya, (i.x.shape[1],  i.x.shape[0]))
            r = self.next(xa, ya)
            return PredictionItem(i.id, r, ya>0.5)
        else:
            r=self.next(i.x,i.y)
            return PredictionItem(i.id,r,i.y)


class WithBackgrounds:
    def __init__(self, ds,bg):
        self.ds=ds
        self.bg=bg
        self.rate=bg.rate

    def __len__(self):
        return len(self.ds)

    def item(self,item,isTrain):
        if not isTrain:
            return self.ds[item]

        return self[item]

    def __getitem__(self, item):
        i=self.ds[item]
        if random.random()>self.rate:
            return self.bg.augment_item(i)
        return i

class NegativeDataSet:
    def __init__(self, path):
        self.path = path

        ldir = os.listdir(path)

        ldir.remove(".DS_Store")

        self.ids = [x[0:x.index('.')] for x in ldir]
        self.exts = [x[x.index('.') + 1:] for x in ldir]

    def __getitem__(self, item):
        in_ext = self.exts[item]

        image = imageio.imread(os.path.join(self.path, self.ids[item] + "." + in_ext))

        out = np.zeros(image.shape)

        if len(out.shape) < 3:
            out = np.expand_dims(out, axis=2)

        out = out.astype(np.float32)

        out = np.sum(out, axis=2)

        out = np.expand_dims(out, axis=2)

        #out = out / np.max(out)

        return PredictionItem(self.ids[item] + str(), image, out)

class BlendedDataSet:
    def __init__(self, child, blendwith, size=(320, 320)):
        self.child = child

        self.blend = blendwith

        self.bids = list(range(len(blendwith)))

        self.size = size

        self.rnd = random.Random(23232)

    def __getitem__(self, item):
        child_item = self.child[item]

        return PredictionItem(child_item.id, self.get_new_image(child_item.x), child_item.y)

    def __len__(self):
        return len(self.child)

    def get_new_image(self, image):
        new_image = cv.resize(image, self.size)

        if self.rnd.choice([True, False]):
            return new_image

        bid = self.rnd.choice(self.bids)
        bland_image = cv.resize(self.blend[bid].x, self.size)

        return cv.addWeighted(new_image, 0.6, bland_image, 0.4, 0)

class SimplePNGMaskDataSet:
    def __init__(self, path, mask, detect_exts=False, in_ext="jpg", out_ext="png"):
        self.path = path
        self.mask = mask
        
        ldir = os.listdir(path)
        
        self.ids=[x[0:x.index('.')] for x in ldir]
        
        self.exts = []
                
        if detect_exts:
            self.exts=[x[x.index('.') + 1:] for x in ldir]
        
        self.detect_exts = detect_exts
        
        self.in_ext = in_ext
        self.out_ext = out_ext
        
        pass

    def __getitem__(self, item):
        in_ext = self.in_ext
        
        if self.detect_exts:
            in_ext = self.exts[item]
        
        out = imageio.imread(os.path.join(self.mask, self.ids[item] + "." + self.out_ext))
        
        if len(out.shape) < 3:
            out = np.expand_dims(out, axis=2)
        
        out = out.astype(np.float32)
        
        out = np.sum(out, axis=2)
        
        out = np.expand_dims(out, axis=2)
        
        out = out / np.max(out)
        
        return PredictionItem(self.ids[item] + str(), imageio.imread(os.path.join(self.path, self.ids[item]+"." + in_ext)), out)

    def isPositive(self, item):
        return True

    def __len__(self):
        return len(self.ids)

NB_WORKERS="auto"
NB_WORKERS_IN_LOADER=1
LOADER_SIZE=50
LOADER_THREADED=True
class KFoldedDataSet:

    def __init__(self,ds,indexes,aug,transforms,folds=5,rs=33,batchSize=16):
        self.ds=ds;
        if aug==None:
            aug=[]
        if transforms==None:
            transforms=[]
        self.aug=aug;

        self.transforms=transforms
        self.batchSize=batchSize
        self.positive={}
        self.kf=ms.KFold(folds,shuffle=True,random_state=rs);
        if hasattr(ds,"folds"):
            self.folds=getattr(ds,"folds");
        else:
            self.folds = [v for v in self.kf.split(indexes)]

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
        for fold in self.folds:
            train = fold[0]
            np.random.shuffle(train)

    def generator(self,foldNum, isTrain=True,negatives="all",returnBatch=False):
        indexes = self.sampledIndexes(foldNum, isTrain, negatives)
        yield from self.generator_from_indexes(indexes,isTrain,returnBatch)

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

    def inner_isPositive(self,x):
        if x in self.positive:
            return self.positive[x]
        self.positive[x]=self.ds.isPositive(x);
        return self.positive[x];

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

    def numBatches(self,fold,negatives,subsample):
        train_indexes = self.sampledIndexes(fold, True, negatives)
        return len(train_indexes)//(round(subsample*self.batchSize))

    def trainOnFold(self,fold:int,model:keras.Model,callbacks=[],numEpochs:int=100,negatives="all",
                    subsample=1.0,validation_negatives=None,verbose=1):
        train_indexes = self.sampledIndexes(fold, True, negatives)
        if validation_negatives==None:
            validation_negatives=negatives
        test_indexes = self.sampledIndexes(fold, False, validation_negatives)

        tl,tg,train_g=self.generator_from_indexes(train_indexes)
        vl,vg,test_g = self.generator_from_indexes(test_indexes,isTrain=False)
        try:
            model.fit_generator(train_g(), len(train_indexes)//(round(subsample*self.batchSize)),
                             epochs=numEpochs,
                             validation_data=test_g(),
                             callbacks=callbacks,
                             verbose=verbose,
                             validation_steps=len(test_indexes)//(round(subsample*self.batchSize)))
        finally:
            tl.terminate();
            tg.terminate();
            vl.terminate();
            vg.terminate();

class KFoldedDataSetImageClassification(KFoldedDataSet):

    def _prepare_vals_from_batch(self, r):
        return np.array(r.images_aug), np.array([x for x in r.data[1]])


class NullTerminatable:

    def terminate(self):
        pass


class NoChangeDataSetImageClassification(KFoldedDataSet):

    def generator_from_indexes(self, indexes,isTrain=True,returnBatch=False):
        m = DataSetLoader(self.ds, indexes, self.batchSize,isTrain=isTrain).generator
        #aug = self.augmentor(isTrain)
        def r():
            num = 0;
            while True:
                for v in m():
                    r = v;
                    x,y= np.array([x for x in r.images]), np.array([x for x in r.data[1]])
                    num=num+1
                    if returnBatch:
                        yield x,y,r
                    else: yield x,y
        return NullTerminatable(),NullTerminatable(),r

class AspectRatioDataSet:
    def __init__(self, child, target_ratio=(1, 1), strategy="center"):
        self.child = child
        self.target_size = target_ratio

        self.strategy = strategy

    def __getitem__(self, item):
        child_item = self.child[item]

        new_size_in = self.get_new_size((child_item.x.shape[0], child_item.x.shape[1]))
        new_size_out = self.get_new_size((child_item.y.shape[0], child_item.y.shape[1]))

        rnd = 0.5;

        if self.strategy == "random":
            rnd = random.random();

        return PredictionItem(child_item.id, self.get_new_image(new_size_in, child_item.x, rnd), self.get_new_image(new_size_out, child_item.y, rnd))

    def __len__(self):
        return len(self.child)

    def get_new_size(self, input_size):
        input_x = input_size[0]
        input_y = input_size[1]

        target_x = self.target_size[1]
        target_y = self.target_size[0]

        input_ratio = input_x / input_y
        output_ratio = target_x / target_y

        if input_ratio > output_ratio:
            input_x = round(input_y * output_ratio)

        elif input_ratio < output_ratio:
            input_y = round(input_x / output_ratio)

        return (input_x, input_y)

    def get_new_image(self, new_size, image, rnd):
        shift_x = 0
        shift_y = 0

        shift = 0

        if new_size[0] != image.shape[0]:
            shift = image.shape[0] - new_size[0]

        elif new_size[1] != image.shape[1]:
            shift = image.shape[1] - new_size[1]

        shift = round(rnd * shift)

        if new_size[0] != image.shape[0]:
            shift_x = shift

        elif new_size[1] != image.shape[1]:
            shift_y = shift

        return image[shift_x:new_size[0] + shift_x, shift_y:new_size[1] + shift_y, :]

class CropAndSplit:
    def __init__(self,orig,n):
        self.ds=orig
        self.parts=n
        self.lastPos=None

    def isPositive(self, item):
        pos = item // (self.parts * self.parts);
        return self.ds.isPositive(pos)

    def __getitem__(self, item):
        pos=item//(self.parts*self.parts);
        off=item%(self.parts*self.parts)
        if pos==self.lastPos:
            dm=self.lastImage
        else:
            dm=self.ds[pos]
            self.lastImage=dm
        row=off//self.parts
        col=off%self.parts
        x,y=dm.x,dm.y
        x1,y1= self.crop(row,col,x),self.crop(row,col,y)
        return PredictionItem(dm.id,x1,y1)

    def crop(self,y,x,image):
        h=image.shape[0]//self.parts
        w = image.shape[1] // self.parts
        return image[h*y:h*(y+1),w*x:w*(x+1), :]

    def __len__(self):
        return len(self.ds)*self.parts*self.parts


class SubDataSet:
    def __init__(self,orig,indexes):
        self.ds=orig
        self.indexes=indexes

    def isPositive(self, item):
        return self.ds.isPositive(self.indexes[item])

    def __getitem__(self, item):
        return self.ds[self.indexes[item]]

    def __len__(self):
        return len(self.indexes)

def split(ds,testSplit,testSplitSeed):
    rn=range(0,len(ds))
    random.seed(testSplitSeed)
    random.shuffle(rn)
    dm=round(1-len(ds)*testSplit)
    return SubDataSet(ds,rn[:dm]),SubDataSet(ds,rn[dm:])

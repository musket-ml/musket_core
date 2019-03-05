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
import scipy
from musket_core import utils

import musket_core.datasources as datasources
import musket_core.dsconfig as dsconfig

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
                batch_x[j].append(r.x[j])
            for j in range(yd):
                batch_y[j].append(r.y[j])
        batch_x=[np.array(x) for x in batch_x]
        batch_y = [np.array(y) for y in batch_y]
        return batch_x,batch_y


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

    def item(self,item,isTrain):
        if not isTrain:
            return self.child[item]

        return self[item]

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


class TextMaskGenerator:
    def __init__(self, textures, band = False):
        self.fonts = [x for x in dir(cv) if x.startswith('FONT_')]

        self.letters = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

        weights = np.ones(len(self.letters))

        weights[0] = 15

        weights = weights / np.sum(weights)

        self.weights = weights

        self.textures = textures

        self.band = band

    def getFont(self):
        return getattr(cv, random.choice(self.fonts))

    def generateText(self, lines, lineLength):
        text = ""

        for lineNum in range(lines):
            line = np.random.choice(self.letters, size=lineLength, p=self.weights)

            text += "".join(line)

            if lineNum == lines - 1:
                continue

            text += "\n"

        return text

    def getLineSize(self, text, font, scale, thickness):
        lines = text.split("\n")

        width = -1

        heights = []
        baselines = []

        for line in lines:
            size = cv.getTextSize(text=line, fontFace=font, fontScale=scale, thickness=thickness)

            if width < size[0][0]:
                width = size[0][0]

            heights.append(size[0][1])
            baselines.append(size[1])

        return width, heights, baselines, lines

    def getInitialMask(self):
        lines = random.randint(1, 2)
        length = random.randint(5, 10)
        thickness = 5
        scale = 3

        text = self.generateText(lines, length)

        font = self.getFont()

        lineWidth, lineHeights, baselines, lines = self.getLineSize(text, font, scale, thickness)

        image = np.zeros((sum(lineHeights) + sum(baselines), lineWidth, 3), np.uint8)

        count = 0

        linePos = 0

        for line in lines:
            lineHeight = lineHeights[count]
            baseLine = baselines[count]

            linePos += lineHeight

            cv.putText(image, line, org=(0, linePos), fontFace=font, fontScale=scale, color=(255,255,255), lineType=cv.LINE_8, thickness=thickness)

            linePos += baseLine

            count += 1

        return image

    def getImageAndMask(self):
        initialMask = []

        if self.band:
            initialMask = np.ones(( random.randint(100, 200),random.randint(500, 1000), 3), np.uint8)
        else:
            initialMask = self.getInitialMask()

        texture = random.choice(self.textures).x.astype(np.uint8)

        maskTexture = initialMask * 0

        baseWidth, baseHeight = self.getTextureBaseSize(texture, initialMask)

        texture = cv.resize(texture, (baseWidth, baseHeight))

        ids = np.indices((initialMask.shape[0], initialMask.shape[1]))

        maskTexture[ids[0], ids[1]] = texture[np.mod(ids[0], baseHeight), np.mod(ids[1], baseWidth)]

        angle = random.randint(-30, 30)

        mask = scipy.ndimage.rotate(initialMask, angle)
        maskTexture = scipy.ndimage.rotate(maskTexture, angle)

        return maskTexture, mask[:, :, 0]

    def getTextureBaseSize(self, texture, mask):
        width = mask.shape[1]
        height = mask.shape[0]

        textureWidth = texture.shape[1]
        textureHeight = texture.shape[0]

        textureAspectRatio = textureWidth / textureHeight
        maskAspectRatio = width / height

        multiplier = 0

        if textureAspectRatio > maskAspectRatio:
            height = width * textureHeight / textureWidth
        else:
            width = height * textureWidth / textureHeight

        return int(width), int(height)

    def __len__(self):
        return 10

    def __getitem__(self, item):
        image, mask = self.getImageAndMask()

        return PredictionItem(str(item), image, mask)


class DropItemsDataset:
    def __init__(self, child, drop_items,times=5):
        self.child = child

        self.drop_items = drop_items

        self.rnd = random.Random(23232)

        self.drop_size = 1

        self.times = times

    def __len__(self):
        return len(self.child)

    def item(self,item,isTrain):
        if not isTrain:
            return self.child[item]

        return self[item]

    def __getitem__(self, item_):
        original_item = self.child[item_]

        input = original_item.x

        mask = self.rescale_mask_to_input(input, original_item.y)

        for time in range(self.times):
            drop_item, drop_mask = self.get_drop_item()

            rescaled_drop_item, rescaled_drop_mask = self.rescale_drop_item(input, drop_item, drop_mask, self.drop_size)

            self.apply_drop_item(input, mask, rescaled_drop_item, rescaled_drop_mask, original_item.id + "_" + str(time))

        return PredictionItem(original_item.id, input, mask.astype(np.bool))

    def apply_drop_item(self, item, mask, drop_item, drop_mask, id=""):
        x = self.rnd.randrange(0, item.shape[1])
        y = self.rnd.randrange(0, item.shape[0])

        self.draw_drop(item, mask, drop_item, drop_mask, x, y, self.rnd.choice(["behind", "above"]), id)

    def draw_drop(self, item, mask, drop_item, drop_mask, x, y, mode="above", id=""):
        half_width = drop_item.shape[1] // 2
        half_height = drop_item.shape[0] // 2

        left = x - half_width
        right = x + half_width

        down = y - half_height
        up = y + half_height

        if left < 0: left = 0
        if down < 0: down = 0

        if up > item.shape[0]: up = item.shape[0]
        if right > item.shape[1]: right = item.shape[1]

        drop_left = left - x + half_width
        drop_right = right - x + half_width

        drop_down = down - y + half_height
        drop_up = up - y + half_height

        temp_mask = mask * 0
        temp_item = item * 0

        temp_mask[down:up, left:right] = drop_mask[drop_down:drop_up,drop_left:drop_right]
        temp_item[down:up, left:right]= drop_item[drop_down:drop_up,drop_left:drop_right]

        temp_mask = np.where(np.sum(temp_mask, 2))

        if mode == "above":
            item[temp_mask] = temp_item[temp_mask]

            mask[temp_mask] = 0
        else:
            old_mask = np.where(np.sum(mask, 2))

            old_item = item * 0

            old_item[old_mask] = item[old_mask] + 0

            item[temp_mask] = temp_item[temp_mask]

            item[old_mask] = old_item[old_mask]

    def rescale_drop_item(self, item, drop_item, drop_mask, scale):
        input_area = item.shape[0] * item.shape[1]

        target_area = scale * input_area

        drop_area = drop_item.shape[0] * drop_item.shape[1]

        sqrt = np.sqrt([target_area / drop_area])[0]

        new_size = (int(sqrt * drop_item.shape[1]), int(sqrt * drop_item.shape[0]))

        new_drop_item = (cv.resize(drop_item / 255, new_size) * 255).astype(np.int32)

        return new_drop_item, self.rescale_mask_to_input(new_drop_item, drop_mask)


    def mask_box_size(self, mask_):
        mask = np.sum(mask_, 2)

        hp = np.sum(mask, 0) > 0
        vp = np.sum(mask, 1) > 0

        return (np.sum(hp), np.sum(vp))

    def rescale_mask_to_input(self, input, mask):
        rescaled_mask = (cv.resize(mask.astype(np.float32), (input.shape[1], input.shape[0])) > 0.5).astype(np.int32)

        rescaled_mask = np.expand_dims(rescaled_mask, 2)

        return rescaled_mask


    def get_drop_item(self):
        drop_item = self.rnd.choice(self.drop_items)

        drop_item_id = drop_item.id

        drop_mask = (cv.resize(drop_item.y, (drop_item.x.shape[1], drop_item.x.shape[0])) > 0.5).astype(np.int32)

        hp = np.sum(drop_mask, 0) > 0
        vp = np.sum(drop_mask, 1) > 0

        hp = np.where(hp)[0]
        vp = np.where(vp)[0]

        drop_mask = np.expand_dims(drop_mask, 2)

        drop_item = drop_item.x * drop_mask

        drop_item = drop_item[vp[0] : vp[-1] + 1, hp[0] : hp[-1] + 1]

        drop_mask = drop_mask[vp[0] : vp[-1] + 1, hp[0] : hp[-1] + 1]

        return drop_item, drop_mask


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
        if stratified:
            self.kf=ms.StratifiedKFold(folds,shuffle=True,random_state=rs)
        self.kf=ms.KFold(folds,shuffle=True,random_state=rs);

        if hasattr(ds,"folds"):
            self.folds=getattr(ds,"folds");
        else:
            if stratified:
                self.folds=[v for v in self.kf.split(indexes,dataset_classes(ds,groupFunc))]
            else: self.folds = [v for v in self.kf.split(indexes)]

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
        self.positive[x]=self.ds.isPositive(x);
        return self.positive[x];

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


class NoChangeDataSetImageClassificationImage(ImageKFoldedDataSet):

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

class DS_Wrapper:
    def __init__(self, name, datasource_cfg, from_directory):
        abs_path = os.path.abspath(from_directory)

        dirname = os.path.dirname(abs_path)

        self.datasource = datasources.GenericDataSource(dsconfig.unpack_config(name, datasource_cfg, dirname))

    def __len__(self):
        return len(self.datasource)

    def __getitem__(self, item):
        ds_item = self.datasource[item]

        return PredictionItem(ds_item.id, ds_item.inputs[0], ds_item.outputs[0])

    def isPositive(self, item):
        return True

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

def split(ds,testSplit,testSplitSeed,stratified=False,groupFunc=None):

    rn=list(range(0,len(ds)))
    if stratified:
        data_classes = dataset_classes(ds, groupFunc)
        vals=ms.StratifiedShuffleSplit(4,testSplit,random_state=testSplitSeed).split(rn,data_classes)
        for v in vals:
            return SubDataSet(ds, v[0]), SubDataSet(ds,v[1])

    random.seed(testSplitSeed)
    random.shuffle(rn)
    dm=round(len(ds)-len(ds)*testSplit)
    return SubDataSet(ds,rn[:dm]),SubDataSet(ds,rn[dm:])


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

class IndicesDistribution:
    def __init__(self, folds, holdout):
        self.folds = folds
        if holdout is not None:
            self.holdout = holdout
        else:
            self.holdout = []

def distribute(ids:list,classes:list,folds:int,seed,extractHoldout:bool):

    for ind1 in range(len(classes)):
        c1 = set(classes[ind1])
        for ind2 in range(c1+1,len(classes)):
            c2 = classes[ind2]
            if len(c1.intersection(c2)) > 0:
                raise ValueError(f'Classes {c1} and {c2} have nonempty intersection')


    actualFolds = folds
    if extractHoldout:
        actualFolds = actualFolds + 1

    positives = []
    negatives = []

    trainIDs = set(ids)
    for cl in classes:
        trainIDs = trainIDs.difference(cl)

    trainIDs = np.array(list(trainIDs),dtype = np.int)

    classIndices = []
    currentSeed = random.getstate()
    random.seed(seed)
    for cl in classes:
        indices = list(range(0, len(cl)))
        random.shuffle(indices)
        shuffled = np.array(cl, dtype=np.int)[indices]
        classIndices.append(shuffled)

    random.seed(currentSeed)

    classFolds = []
    for ci in classIndices:
        _folds = np.array_split(ci, actualFolds)
        classFolds.append(_folds)

    actualValidationFolds = []
    for ind in range(0, actualFolds):
        avf = np.array([],dtype=np.int)
        for cf in classFolds:
            avf = np.concatenate((avf, ci))
        actualValidationFolds.append(avf)

    for ind1 in range(0, actualFolds):
        f1 = actualValidationFolds[ind1]
        for ind2 in range(ind1+1, actualFolds):
            f2 = actualValidationFolds[ind2]
            if len(set(f1).intersection(f2)) != 0:
                raise ValueError(f'Validation folds {f1} and {f2} have nonempty intersection')

    distributionsCount = 1
    distributions = []
    if extractHoldout:
        distributionsCount = actualFolds

    for dInd in range(0,distributionsCount):

        foldsValidation = actualValidationFolds.copy()
        holdout = None
        if extractHoldout:
            holdout = foldsValidation.pop(dInd)

        foldsTrain = (np.zeros((folds, len(trainIDs)), dtype = np.int) + np.array(list(trainIDs), dtype = np.int)).tolist()
        for shift in range(1, folds):
            perm = np.roll(foldsValidation, shift, axis = 0)
            for ind in range(0, folds):
                foldsTrain[ind] = np.concatenate((foldsTrain[ind], perm[ind]))

        for ind1 in range(0, folds):
            f1 = foldsTrain[ind1]
            for ind2 in range(ind1+1, folds):
                f2 = foldsTrain[ind2]
                expectedLength = len(trainIDs)
                for ind3 in range(0, folds):
                    if ind3 == ind1 or ind3 == ind2:
                        continue
                    expectedLength = expectedLength + len(foldsValidation[ind3])
                actualLength = len(set(f1).intersection(f2))
                if actualLength != expectedLength:
                    raise ValueError(f'Train folds {f1} and {f2} have {actualLength} while it should be {expectedLength}')

        foldsList = []
        for ind in range(0, folds):
            foldsList.append((foldsTrain[ind],foldsValidation[ind]))

        distribution = IndicesDistribution(foldsList,holdout)
        distributions.append(distribution)

    if extractHoldout:
        for dInd in range(0,len(distributions)):
            distrib = distributions[dInd]
            dHoldout = set(distribution.holdout.tolist())
            dFolds = distribution.folds
            for fInd in range(0,len(dFolds)):
                train, validation = dFolds[fInd]
                if len(dHoldout.intersection(train.tolist())) != 0:
                    raise ValueError(f'Holdout {dInd} and train {fInd} have nonempty intersection')
                if len(dHoldout.intersection(validation.tolist())) != 0:
                    raise ValueError(f'Holdout {dInd} and validation {fInd} have nonempty intersection')

    return distributions

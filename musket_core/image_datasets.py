import imageio
import cv2 as cv

from musket_core import datasources as datasources, dsconfig as dsconfig
from musket_core.datasets import PredictionItem, ImageKFoldedDataSet, DataSetLoader, NullTerminatable
import os
import  numpy as np
import random
import scipy
import imgaug

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
    
    def get_train_item(self,item):
        return self[item]

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

    def item_by_id(self, id):
        item = self.datasource.ids.index(id)

        return self[item]

    def isPositive(self, item):
        return True


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
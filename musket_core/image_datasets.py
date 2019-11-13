import imageio
import cv2 as cv
import pandas as pd
from musket_core import datasources as datasources, dsconfig as dsconfig
from musket_core.datasets import PredictionItem, ImageKFoldedDataSet, DataSetLoader, NullTerminatable,DataSet
from musket_core import context
import os
import  numpy as np
import random
import scipy
import tqdm
import imgaug
import math
from musket_core.coders import classes_from_vals,rle2mask_relative,mask2rle_relative,rle_decode,rle_encode,\
    classes_from_vals_with_sep


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
            self.lastPos=pos
            self.lastImage=dm
        row=off//self.parts
        col=off%self.parts
        x,y=dm.x,dm.y
        x1,y1= self.crop(row,col,x),self.crop(row,col,y)

        vs=PredictionItem(dm.id,x1,y1)
        if hasattr(dm, "prediction" ) and dm.prediction is not None:
            pred=self.crop(row,col,dm.prediction)
            vs.prediction=pred
        vs.imageId=dm.id
        vs.row=row
        vs.col=col
        return vs

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
    
    









class AbstractImagePathDataSet(DataSet):
    
    def __init__(self,imagePath):
        self.images={}
        if imagePath is None:
            return;
        if isinstance(imagePath, list): 
            for v in imagePath:
                self.addPath(v)
        else: 
            self.addPath(imagePath)
        self.dim=3 

    def addPath(self, imagePath):
        p0 = os.path.join(context.get_current_project_data_path(), imagePath)
        if not os.path.exists(p0):
            p0 = imagePath
        ld0 = os.listdir(p0)
        for x in ld0:
            fp = os.path.join(p0, x)
            self.images[x] = fp
            self.images[x[:-4]] = fp
        
    
    def get_value(self,im_id):
        im=imageio.imread(self.images[im_id])
        if len(im.shape)!=3:
            im=np.expand_dims(im, -1)
        if im.shape[2]!=self.dim:
            if self.dim==3:
                im=np.concatenate([im,im,im],axis=2)
            elif self.dim==1:         
                im=np.mean(im,axis=2)
            else:
                raise ValueError("Unsupported conversion")    
        return im 
        
    def __getitem__(self, item)->PredictionItem:
        raise ValueError()
    
class CSVReferencedDataSet(AbstractImagePathDataSet):        
    
    def readCSV(self,csvPath):
        try:
            self.data=pd.read_csv(os.path.join(context.get_current_project_data_path(), csvPath))
        except:
            try:
                self.data=pd.read_csv(os.path.join(context.get_current_project_data_path(), csvPath),encoding="cp1251")
            except:    
                self.data=pd.read_csv(csvPath)

    def ordered_vals(self, imColumn):
        return sorted(list(set(self.get_values(imColumn))))

    def __init__(self,imagePath,csvPath,imColumn):
        super().__init__(imagePath)
        self.imColumn=imColumn
        if isinstance(csvPath, str):
            self.readCSV(csvPath)
        else:
            self.data=csvPath    
        self.splitColumns={}
        for m in self.data.columns:
            parts=m.split("_")
            ind=0
            for col in parts:                
                if not col in self.data.columns:
                    try:
                        vl=[x[ind] for x in self.data[m].str.split("_")]
                        self.data.insert(0,col,value=vl)
                        self.splitColumns[col]=m
                    except:
                        pass    
                ind=ind+1
        self.imageIds=self.ordered_vals(imColumn)                            
    
    def _id(self,item):
        imageId=self.imageIds[item]
        return imageId
    
    def get_values(self,col):
        return self.data[col]
            
    def __len__(self):
        return len(self.imageIds)
    
    def get_all_about(self,item):
        return self.data[self.data[self.imColumn]==item]
    
    def __getitem__(self, item)->PredictionItem:
        raise ValueError()
    
    def _encode_template(self,template_id,template,val):
        rs=[]
        for q in template:
            v=val[q]
            rs.append(v)
            del val[q]
        val[template_id]="_".join(rs)
        return val    
    
    def _recode(self,seq):
        
        templates={}
        for q in self.splitColumns:
            r=self.splitColumns[q]
            templates[r]=r.split("_")
            
        for item in seq:            
            for t in templates:
                self._encode_template(t,templates[t],item)
        return seq


class BinarySegmentationDataSet(CSVReferencedDataSet):    
    
    def __init__(self,imagePath,csvPath,imColumn,rleColumn=None,maskShape=None,rMask=True,isRel=False):   
        super().__init__(imagePath,csvPath,imColumn)
        self.rleColumn=rleColumn
        self.maskShape=maskShape
        self.rMask=rMask
        self.rle_decode=rle_decode
        self.rle_encode=rle_encode
        if isRel:
            self.rle_decode=rle2mask_relative
            self.rle_encode=mask2rle_relative
        
    def get_target(self,item):    
        imageId=self.imageIds[item]
        vl = self.get_all_about(imageId)
        rleString = vl[self.rleColumn].values[0]
        if isinstance(rleString, str):
            if rleString.strip() != "-1" and len(rleString.strip())>0:
                return 1
        return 0   
    
    def isPositive(self,item):
        return self.get_target(item)==True 
    
    def get_rleString(self, item):
        imageId=item.id
        vl = self.get_all_about(imageId)
        rleString = vl[self.rleColumn].values[0]
        if isinstance(rleString,float):
            if math.isnan(rleString):
                return ""
        return rleString
        
    def get_mask(self, image,imShape):
        prediction = None
        vl = self.get_all_about(image)
        rleString = vl[self.rleColumn].values[0]
        if isinstance(rleString, str):
            if rleString.strip() != "-1":
                shape = (imShape[0], imShape[1])
                if self.maskShape is not None:
                    shape = self.maskShape
                if self.rMask:
                    prediction = self.rle_decode(rleString, (shape[1],shape[0]))
                else:
                    prediction = self.rle_decode(rleString, shape)
                
                prediction=np.rot90(prediction)
                prediction=np.flipud(prediction)
                prediction = np.expand_dims(prediction,2).astype(np.bool)
                
        if prediction is None:
            prediction = np.zeros((imShape[0], imShape[1], 1), dtype=np.bool)
        return prediction
    
    
    
    def __getitem__(self, item)->PredictionItem:
        imageId=self.imageIds[item]
        image=self.get_value(imageId)
        prediction = self.get_mask(imageId,image.shape)
        return PredictionItem(self._id(item),image,prediction)
    

    def _to_rle(self, o):
        o = np.flipud(o)
        o = np.rot90(o, -1)
        rle = self.rle_encode(o)
        return rle

    def encode(self,item:PredictionItem,encode_y=False,treshold=0.5):
        if isinstance(item, PredictionItem):
            imageId=item.id
            if encode_y:
                o=item.y
            else:    
                o=item.prediction
            if (o.dtype!=np.bool):
                    o=o>treshold    
            rle = self._to_rle(o)
            return { self.imColumn:imageId,self.rleColumn:rle}        
        if isinstance(item, DataSet):
            res=[]            
            for i in tqdm.tqdm(range(len(item)),"Encoding dataset"):
                q=item[i]
                res.append(self.encode(q,encode_y,treshold))                
            return pd.DataFrame(res,columns=[self.imColumn,self.rleColumn])     

class MultiClassSegmentationDataSet(BinarySegmentationDataSet):    
    
    def __init__(self,imagePath,csvPath,imColumn,rleColumn,clazzColumn,maskShape=None,rMask=True,isRel=False):   
        super().__init__(imagePath,csvPath,imColumn,rleColumn,maskShape,rMask,isRel)
        self.clazzColumn=clazzColumn    
        self.classes=sorted(list(set(self.data[clazzColumn].values)))
        self.class2Num={}
        self.num2class={}
        num=0
        for c in self.classes:
            self.class2Num[c]=num
            self.num2class[num]=c
            num=num+1
            
    def encode(self,item:PredictionItem,encode_y=False,treshold=0.5):
        if isinstance(item, PredictionItem):
            raise NotImplementedError("Multiclass segmentation is only capable to encode datasets")       
        if isinstance(item, DataSet):
            res=[]            
            for i in tqdm.tqdm(range(len(item)),"Encoding dataset"):
                q=item[i]
                imageId=q.id
                for j in range(len(self.classes)):
                    if encode_y:
                        vl=q.y[:,:,j:j+1]>treshold
                    else:
                        vl=q.prediction[:,:,j:j+1]>treshold
                    rle=self._to_rle(vl)
                    res.append({ self.imColumn:imageId,self.rleColumn:rle,self.clazzColumn:self.num2class[j]})
            res=self._recode(res)
                    
            clns=[]
            for c in self.splitColumns:
                if not self.splitColumns[c] in clns:
                    clns.append(self.splitColumns[c])
            r=[self.imColumn,self.clazzColumn,self.rleColumn]
            for c in r:
                if not c in self.splitColumns:
                    clns.append(c)
            return pd.DataFrame(res,columns=clns)            
        
    def get_target(self,item):    
        imageId=self.imageIds[item]
        vl = self.get_all_about(imageId)
        for i in range(len(vl)):
            rleString = vl[self.rleColumn].values[i]
            if isinstance(rleString, str):
                if rleString.strip() != "-1":
                    return 1
        return 0    
    
    
    def get_mask(self, image,imShape):
        prediction = np.zeros((imShape[0], imShape[1], len(self.classes)), dtype=np.bool)
        vl = self.get_all_about(image)
        rle=vl[self.rleColumn].values
        classes=vl[self.clazzColumn].values
        for i in range(len(vl)):
            rleString = rle[i]
            clazz=classes[i]
            if isinstance(rleString, str):
                if rleString.strip() != "-1":
                    shape = (imShape[0], imShape[1])
                    if self.maskShape is not None:
                        shape = self.maskShape
                    if self.rMask:     
                        lp = self.rle_decode(rleString, (shape[1],shape[0]))
                        
                    else:
                        lp = self.rle_decode(rleString, shape)
                    lp=np.rot90(lp)
                    lp=np.flipud(lp)        
                    prediction[:,:,self.class2Num[clazz]]=lp        
        return prediction
    
    def __getitem__(self, item)->PredictionItem:
        imageId=self.imageIds[item]
        image=self.get_value(imageId)
        prediction = self.get_mask(imageId,image.shape)
        return PredictionItem(imageId,image,prediction)


class InstanceSegmentationDataSet(MultiClassSegmentationDataSet):

    def __init__(self, imagePath, csvPath, imColumn, rleColumn, clazzColumn, maskShape=None, rMask=True, isRel=False):
        super().__init__(imagePath,csvPath,imColumn,rleColumn,clazzColumn,maskShape,rMask,isRel)

        rawClasses = self.data[clazzColumn].values
        def refineClass(x):
            if "_" in str(x):
                return x[:x.index("_")]
            else:
                return x

        self.classes = sorted(list(set([ refineClass(x) for x in rawClasses ])))
        self.class2Num = {}
        self.num2class = {}
        num = 0
        for c in self.classes:
            self.class2Num[c] = num
            self.num2class[num] = c
            num = num + 1

    def meta(self):
        return { 'CLASSES': self.classes }

    def get_mask(self, image, imShape):
        prediction = []
        vl = self.get_all_about(image)
        rle = vl[self.rleColumn].values
        classes = vl[self.clazzColumn].values
        for i in range(len(vl)):
            rleString = rle[i]
            clazz = classes[i]
            if "_" in str(clazz):
                clazz = clazz[:clazz.index("_")]
            if isinstance(rleString, str):
                if rleString.strip() != "-1":
                    shape = (imShape[0], imShape[1])
                    if self.maskShape is not None:
                        shape = self.maskShape
                    if self.rMask:
                        lp = self.rle_decode(rleString, (shape[1], shape[0]))

                    else:
                        lp = self.rle_decode(rleString, shape)
                    lp = np.rot90(lp)
                    lp = np.flipud(lp)
                    prediction.append((lp, int(clazz)))
        return prediction

    def encode(self, item, encode_y=False, treshold=0.5):
        if isinstance(item, PredictionItem):
            raise NotImplementedError("Instance segmentation is only capable to encode datasets")
        if isinstance(item, DataSet):
            res = []
            for i in tqdm.tqdm(range(len(item)), "Encoding dataset"):
                q = item[i]
                imageId = q.id
                for j in range(len(self.classes)):
                    if encode_y:
                        vl = q.y[:, :, j:j + 1] > treshold
                    else:
                        vl = q.prediction[:, :, j:j + 1] > treshold
                    labels = vl[0]
                    masks = vl[2]
                    if len(labels) != len(masks):
                        raise Exception(f"{imageId} does not have same ammount of masks and labels")
                    for i in range(len(masks)):
                        mask = masks[i]
                        label = labels[i]
                        rle = self._to_rle(mask)
                        res.append({self.imColumn: imageId, self.rleColumn: rle, self.clazzColumn: label})
            res = self._recode(res)

            clns = []
            for c in self.splitColumns:
                if not self.splitColumns[c] in clns:
                    clns.append(self.splitColumns[c])
            r = [self.imColumn, self.clazzColumn, self.rleColumn]
            for c in r:
                if not c in self.splitColumns:
                    clns.append(c)
            return pd.DataFrame(res, columns=clns)

    def __getitem__(self, item)->PredictionItem:
        imageId=self.imageIds[item]
        image=self.get_value(imageId)
        gt = self.get_mask(imageId,image.shape)

        labels = []
        masks = []
        bboxes = []
        for m in gt:
            mask = m[0]
            if np.max(mask) == 0:
                continue
            label = m[1]
            labels.append(label)
            masks.append(mask > 0)
            bboxes.append(getBB(mask, True))

        labelsArr = np.array(labels, dtype=np.int64) + 1
        bboxesArr = np.array(bboxes, dtype=np.float32)
        masksArr = np.array(masks, dtype=np.int16)

        y = (labelsArr, bboxesArr, masksArr)
        return PredictionItem(imageId,image,y)

class BinaryClassificationDataSet(CSVReferencedDataSet):
    
    def ordered_vals(self, imColumn):
        return self.data[imColumn].values

    def initClasses(self, clazzColumn):
        self.hasNa=self.data[clazzColumn].isna().sum()>0        
        vals=self.data[clazzColumn][self.data[clazzColumn].notna()].values
        if isinstance(vals[0],str):
            return sorted(list(set([x.strip() for x in set(vals)])))
        ss=set(vals)
        if self.hasNa:
            return [""]+list(ss)
        return sorted(list(ss))

    def __init__(self,imagePath,csvPath,imColumn,clazzColumn):   
        super().__init__(imagePath,csvPath,imColumn)
        self.classes=self.initClasses(clazzColumn)
        self.class2Num={}
        self.num2Class={}
        self.clazzColumn=clazzColumn        
        num=0
        for c in self.classes:
            self.class2Num[c]=num
            self.num2Class[num]=c
            num=num+1            
            
    def get_target(self,item):    
        imageId=self.imageIds[item]
        vl = self.get_all_about(imageId)        
        result=np.zeros((1),dtype=np.bool)
        for i in range(len(vl)):
            clazz = vl[self.clazzColumn].values[i]
            if isinstance(clazz,str):
                clazz=clazz.strip()
            if clazz in self.class2Num and self.class2Num[clazz]==1:
                result[0]=1                
        return result        
            
    def __getitem__(self, item)->PredictionItem:
        imageId=self.imageIds[item]
        image=self.get_value(imageId)
        prediction = self.get_target(item)
        return PredictionItem(self._id(item),image,prediction)
    
    def _encode_class(self,o,treshold=0.5):
        o=o>treshold
        if o[0]:
            return self.num2Class[1]
        return self.num2Class[0]
    
    

    def _encode_x(self, item):
        return item.id
    
    def _encode_item(self, item:PredictionItem, encode_y=False, treshold=0.5):
        imageId=self._encode_x(item)
        if encode_y:
            o=item.y
        else:    
            o=item.prediction
        return { self.imColumn:imageId,self.clazzColumn:self._encode_class(o,treshold)}
        

    def encode(self,item:PredictionItem,encode_y=False,treshold=0.5):
        if isinstance(item, PredictionItem):
            return self._encode_item(item, encode_y, treshold)            
        if isinstance(item, DataSet):
            res=[]            
            for i in tqdm.tqdm(range(len(item)),"Encoding dataset"):
                q=item[i]
                res.append(self.encode(q,encode_y,treshold))                
            return self._create_dataframe(res)  
        
    def _create_dataframe(self, items):
        return pd.DataFrame(items,columns=[self.imColumn,self.clazzColumn])
            
    
class CategoryClassificationDataSet(BinaryClassificationDataSet): 
    
    def __init__(self,imagePath,csvPath,imColumn,clazzColumn):   
        super().__init__(imagePath,csvPath,imColumn,clazzColumn)
        
    def _encode_class(self,o,treshold=0.5):
        return self.num2Class[(np.where(o==o.max()))[0][0]]           
            
    def get_target(self,item):    
        imageId=self.imageIds[item]
        vl = self.get_all_about(imageId)        
        result=np.zeros((len(self.classes)),dtype=np.bool)                
        for i in range(len(vl)):
            clazz = vl[self.clazzColumn].values[i]
            if isinstance(clazz, str):
                clazz=clazz.strip()
            result[self.class2Num[clazz]]=1            
        return result

import math

class FolderClassificationDataSet(CategoryClassificationDataSet):
    def __init__(self,imagePath,folder,imColumn,clazzColumn):
        self.images={}
        self.data={"ImageId":[],"Clazz":[]} 
        for x in imagePath:
            self.addPath(os.path.join(folder,x))
        tmp=self.images
            
        super().__init__([],pd.DataFrame(self.data),imColumn,clazzColumn)
        self.images=tmp
    
    def addPath(self, imagePath):
        p0 = os.path.join(context.get_current_project_data_path(), imagePath)
        if not os.path.exists(p0):
            p0 = imagePath
        ld0 = os.listdir(p0)
        nm=os.path.basename(p0)
        for x in ld0:
            ext=x[-4:]
            if ext==".jpg" or ext==".png" or ext==".gif":
                fp = os.path.join(p0, x)
                self.images[x] = fp
                self.data["ImageId"].append(x)
                self.data["Clazz"].append(nm)
                self.images[x[:-4]] = fp
                

class FolderDataSet(FolderClassificationDataSet):
    def __init__(self,imagePath,folder,imColumn,clazzColumn):
        self.images={}
        self.data={"ImageId":[],"Clazz":[]}         
        self.addPath(folder)
        tmp=self.images            
        CategoryClassificationDataSet.__init__(self,[],pd.DataFrame(self.data),imColumn,clazzColumn)
        self.images=tmp
        
    

def _join_column_get_trg(trg,values):
    def func(item):
        v=values[item]
        return v
    return func   
    
def _enc(trg):    
    def _encode_item(item:PredictionItem, encode_y=False, treshold=0.5):
        
        imageId=trg._encode_x(item)
        res={trg.imColumn:imageId}
        if encode_y:
            o=item.y
        else:
            o=item.prediction
        
        o=o>treshold        
        for i in range(len(trg.classes)):
            if trg.classes[i] in trg.columnCls:
                classes=trg.columnCls[trg.classes[i]]
                if o[i]:
                    res[trg.classes[i]]=classes[1]
                else:
                    res[trg.classes[i]]=classes[0]
            else: 
                res[trg.classes[i]]=o[i]
        return res
    return _encode_item

def _to_int(vls):
    mn=sorted(list(set(vls)))
    return vls==mn[1],mn
            
class MultiClassClassificationDataSet(BinaryClassificationDataSet): 
    
    
    def initClasses(self, clazzColumn):
        if clazzColumn not in self.data.columns:
            cls=clazzColumn.split("|")
            if len(cls)>1:
                allVls=[]
                self.columnCls={}
                for c in cls:
                    vls = self.data[c].values
                    if not "int" in str(vls.dtype) and not "bool" in str(vls.dtype):
                        self.data[c]=self.data[c].fillna("")
                        vls,column_cls=_to_int(self.data[c].str.strip())
                        self.columnCls[c]=column_cls
                    allVls.append(vls)
                
                self.allValues=np.stack(allVls,axis=1)
                self.get_target=_join_column_get_trg(self,self.allValues)
                self._encode_item=_enc(self)
                def _cr(items):
                    return pd.DataFrame(items,columns=tuple([self.imColumn]+self.classes))
                self._create_dataframe=_cr
                return cls                 
        tc=self.data[clazzColumn].values
        clz,sep=classes_from_vals_with_sep(tc)
        self.sep=sep
        return clz
        
    
    def _encode_class(self,o,treshold):
        o=o>treshold
        res=[]
        for i in range(len(o)):
            if o[i]==True:
                res.append(self.num2Class[i])
        if self.sep is None:
            if len(res)==0:
                return ""
            return res[0]        
        return self.sep.join(res)            
    
    def __init__(self,imagePath,csvPath,imColumn,clazzColumn):   
        super().__init__(imagePath,csvPath,imColumn,clazzColumn)                
            
    def get_target(self,item):    
        imageId=self.imageIds[item]
        vl = self.get_all_about(imageId)        
        result=np.zeros((len(self.classes)),dtype=np.bool)                
        for i in range(len(vl)):
            
            clazz = vl[self.clazzColumn].values[i]
            
            if isinstance(clazz, float):
                if math.isnan(clazz):
                    continue
            clazz=clazz.strip()    
            if len(clazz)==0:
                continue
            if self.sep is not None:
                for w in clazz.split(self.sep):
                    result[self.class2Num[w]]=1
            else:
                result[self.class2Num[clazz]]=1        
                        
        return result



class MultiOutputClassClassificationDataSet(MultiClassClassificationDataSet):

    def __init__(self,imagePath,csvPath,imColumn,clazzColumns):
        super().__init__(imagePath,csvPath,imColumn,clazzColumns[0])
        self.classes=[]
        self.class2Num=[]
        self.num2Class=[]
        self.clazzColumns=clazzColumns
        for clazzColumn in clazzColumns:
            cls=self.initClasses(clazzColumn)
            self.classes.append(cls)
            class2Num={}
            num2Class={}
            num=0
            for c in cls:
                class2Num[c]=num
                num2Class[num]=c
                num=num+1
            self.class2Num.append(class2Num)
            self.num2Class.append(num2Class)

    def get_target(self,item):
        imageId=self.imageIds[item]
        vl = self.get_all_about(imageId)
        num=0
        results=[]
        for clazzColumn in self.clazzColumns:
            result=np.zeros((len(self.classes[num])),dtype=np.bool)
            for i in range(len(vl)):

                clazz = vl[clazzColumn].values[i]
                if isinstance(clazz, float):
                    if math.isnan(clazz):
                        continue
                if len(clazz.strip())==0:
                    continue
                if " " in clazz:
                    for w in clazz.split(" "):
                        result[self.class2Num[num][w]]=1
                elif "|" in clazz:
                    for w in clazz.split("|"):
                        result[self.class2Num[num][w]]=1
                else:
                    result[self.class2Num[num][clazz.strip()]]=1
            results.append(result)
            num=num+1
        return results

def getBB(mask,reverse=False):
    a = np.where(mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    minX = max(0, bbox[0] - 10)
    maxX = min(mask.shape[0], bbox[1] + 1 + 10)
    minY = max(0, bbox[2] - 10)
    maxY = min(mask.shape[1], bbox[3] + 1 + 10)
    if reverse:
        return np.array([minY, minX, maxY, maxX])
    else:
        return np.array([minX, minY, maxX, maxY])
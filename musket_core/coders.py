import numpy as np
import math
from musket_core import context
import os
import imageio

coders={}

def coder(name): 
    def inner(func): 
        func.coder=True
        coders[name]=func
        return func        
    return inner #this is the fun_obj mentioned in the above content

def classes_from_vals_with_sep(tc,sep=" |_",emptyIs=None):
    realC=set()
    hasMulti=False
    nan=None
    usedSep=None
    for v in set(tc):
            if isinstance(v, float):
                if math.isnan(v):
                    nan=v
                    continue
            bs=False    
            if isinstance(v, str):
                v=v.strip()    
                if len(v)==0:
                    if emptyIs is None:
                        continue
                    else:
                        realC.add(v)
                        continue
                if usedSep is None:     
                    for s in sep:
                        if s in v:
                            bs=True
                            usedSep=s
                            hasMulti=True
                            for w in v.split(s):
                                realC.add(w.strip())
                else:
                    if usedSep in v:
                            bs=True
                            
                            hasMulti=True
                            for w in v.split(usedSep):
                                realC.add(w.strip())                  
            if not bs:
                realC.add(v)
    realC=sorted(list(realC))            
    if nan is not None and not hasMulti:
        realC.append("nan")                                      
    return realC,usedSep 

def classes_from_vals(tc,sep=" |_",emptyIs=None):
    cl,sep=classes_from_vals_with_sep(tc, sep, emptyIs);
    return cl

@coder("number")
class NumCoder:
    def __init__(self,vals,ctx):
        self.values=vals
        self.ctx=ctx
    def __getitem__(self, item):
        return np.array([self.values[item]])
    def _decode(self,item,tr=0.5):
        return item[0]
    
@coder("normalized_number")
class NormalziedNumCoder:
    def __init__(self,vals,ctx):
        self.values=np.nan_to_num(vals)
        self.values=(self.values-np.min(self.values))/np.max(self.values)-np.min(self.values)
        self.ctx=ctx
    def __getitem__(self, item):
        return np.array([self.values[item]])
    def _decode(self,item,tr=0.5):
        return item[0]    
    
      
    
class ConcatCoder:       
    def __init__(self,coders):
        self.coders=coders
    def __getitem__(self, item):
        c=[i[item] for i in self.coders]
        return np.concatenate(c,axis=0)
    
    def _decode(self,item,tr=0.5):
        raise NotImplementedError("Does not implemented yet") 
        
@coder("multi_class")        
class ClassCoder:
    

    def initClasses(self, vals, sep):
        classes,sep= classes_from_vals_with_sep(vals, sep)
        self.sep=sep
        return classes
    
    def init_from_meta(self,data:dict):
        self.class2Num=data
        self.num2Class={}
        for c in data:
            self.num2Class[data[c]]=c
        self.classes=data.keys()
        self.sep=" "    
        
    def __init__(self,vals,ctx,sep="|_ ",cat=False):
        self.class2Num={}
        self.ctx=ctx
        self.num2Class={}
        self.values=vals
        cls=self.initClasses(vals, sep)
        self.classes=cls
        num=0        
        for c in cls:
            self.class2Num[c]=num
            if c=="nan":
                self.num2Class[num]=math.nan
            else:    
                self.num2Class[num]=c
            num=num+1
    
    def __getitem__(self, item):
        return self.encode(self.values[item])        
    
    def _decode(self,o,treshold=0.5):
        o=o>treshold
        res=[]
        for i in range(len(o)):
            if o[i]==True:
                res.append(self.num2Class[i])
        if len(res)==1:
            return res[0]
        
        if self.sep is None:
            if len(res)>0:
                return res[0]
            return ""                        
        return self.sep.join(res)         
            
    def encode(self,clazz):            
        result=np.zeros((len(self.classes)),dtype=np.bool)
        
        if isinstance(clazz, str):
            clazz=clazz.strip()
            if len(clazz)==0:
                if clazz in self.class2Num:
                    result[self.class2Num[clazz]]=1
                return result
            bs=False
            if self.sep is not None:
                if self.sep in clazz:
                    bs=True
                    for w in clazz.split(self.sep):
                        result[self.class2Num[w]]=1                        
            if not bs:
                result[self.class2Num[clazz]]=1
        else:
            if math.isnan(clazz) and not clazz  in self.class2Num:
                return result
            
            result[self.class2Num[clazz]]=1
                        
        return result

@coder("categorical_one_hot")    
class CatClassCoder(ClassCoder):
    
    def initClasses(self, vals, sep):
        return classes_from_vals(vals, "",emptyIs=True)
    
    def _decode(self,o,tr=0.5): 
        return self.num2Class[(np.where(o==o.max()))[0][0]]
    
    def encode(self,clazz):            
        result=np.zeros((len(self.classes)),dtype=np.bool)
        
        if isinstance(clazz, str):
            clazz=clazz.strip()
        if isinstance(clazz, float):
            if math.isnan(clazz):
                clazz="nan"    
            
        result[self.class2Num[clazz]]=1
                        
        return result

@coder("binary")    
class BinaryClassCoder(ClassCoder):
    
    def _decode(self,o,treshold=0.5):
        o=o>treshold
        if o[0]:
            return self.num2Class[1]
        return self.num2Class[0] 
    
    def initClasses(self, vals, sep):
        if len(vals)==1 and vals[0]=='':
            vals=[False,True]
        return classes_from_vals(vals, "",emptyIs=True)
    
    def encode(self,clazz):            
        result=np.zeros(1,dtype=np.bool)
        
        if isinstance(clazz, str):
            if len(clazz.strip())==0:
                return result
            if self.class2Num[clazz.strip()]==1:
                result[0]=1
        else:
            if math.isnan(clazz) and not clazz  in self.class2Num:
                return result
            if self.class2Num[clazz]==1:
                result[0]=1
                        
        return result    

def mask2rle_relative(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask_relative(rle, shape):
    width=shape[0]
    height=shape[1]
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    if isinstance(mask_rle, float):
        return np.zeros(shape,dtype=np.bool)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
        
@coder("rle")        
class RLECoder:
    
    def __init__(self,values,ctx):
        self.values=values
        self.ctx=ctx
        self.inited=False
    
    def __getitem__(self, item):
        if not self.inited:
            self.init()            
        return rle_decode(self.values[item],self.shape)
    
    def _decode(self,item,tr):
        return rle_encode(item>tr)
    
    def init(self):
        f=self.ctx.imageCoders[0]
        fi=f[0]
        self.shape=(fi.shape[0],fi.shape[1])    
                
@coder("image")        
class ImageCoder:
    
    def __init__(self,values,ctx):
        self.images={}
        imagePath=ctx.imagePath
        if imagePath is None:
            return;
        if isinstance(imagePath, list): 
            for v in imagePath:
                self.addPath(v)
        else: 
            self.addPath(imagePath)
        self.dim=3
        self.values=values 

    def addPath(self, imagePath):
        p0 = os.path.join(context.get_current_project_data_path(), imagePath)
        if not os.path.exists(p0):
            p0 = imagePath
        ld0 = os.listdir(p0)
        for x in ld0:
            fp = os.path.join(p0, x)
            self.images[x] = fp
            self.images[x[:-4]] = fp
    
    def __getitem__(self, item):
        return self.get_value(self.values[item])    
    
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


def get_coder(name:str,ctx,parent):
    if name in coders:
        return coders[name](ctx,parent)
    raise ValueError("Unknown coder:"+name)


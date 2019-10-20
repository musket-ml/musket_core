import numpy as np
import math
from musket_core import context
import os
import imageio

def classes_from_vals(tc,sep=" |"):
    realC=set()
    hasMulti=False
    nan=None
    for v in set(tc):
            if isinstance(v, float):
                if math.isnan(v):
                    nan=v
                    continue
            bs=False    
            if isinstance(v, str):
                v=v.strip()    
                if len(v)==0:
                    continue
                for s in sep:
                    if s in v:
                        bs=True
                        hasMulti=True
                        for w in v.split(s):
                            realC.add(w.strip())  
            if not bs:
                realC.add(v)
    realC=sorted(list(realC))            
    if nan is not None and not hasMulti:
        realC.append(nan)                                      
    return realC 

class NumCoder:
    def __init__(self,vals):
        self.values=vals
    def __getitem__(self, item):
        return np.array([self.values[item]])
    def _decode_class(self,item):
        return item[0] 
    
class ConcatCoder:       
    def __init__(self,coders):
        self.coders=coders
    def __getitem__(self, item):
        c=[i[item] for i in self.coders]
        return np.concatenate(c,axis=0)
    
    def _decode_class(self,item):
        raise NotImplementedError("Does not implemented yet") 
        
class ClassCoder:
    
    def __init__(self,vals,sep=" |",cat=False):
        self.class2Num={}
        self.num2Class={}
        self.values=vals
        cls=classes_from_vals(vals,sep);
        self.classes=cls
        num=0
        self.sep=sep
        for c in cls:
            self.class2Num[c]=num
            self.num2Class[num]=c
            num=num+1
    
    def __getitem__(self, item):
        return self.encode(self.values[item])        
    
    def _decode_class(self,o,treshold=0.5):
        o=o>treshold
        res=[]
        for i in range(len(o)):
            if o[i]==True:
                res.append(self.num2Class[i])                
        return self.sep[0].join(res)         
            
    def encode(self,clazz):            
        result=np.zeros((len(self.classes)),dtype=np.bool)
        
        if isinstance(clazz, str):
            if len(clazz.strip())==0:
                return
            bs=False
            for s in self.sep:
                if s in clazz:
                    bs=True
                    for w in clazz.split(s):
                        result[self.class2Num[w]]=1                        
            if not bs:
                result[self.class2Num[clazz.strip()]]=1
        else:
            if math.isnan(clazz) and not clazz  in self.class2Num:
                return result
            
            result[self.class2Num[clazz]]=1
                        
        return result
    
class CatClassCoder(ClassCoder):
    
    def _encode_class(self,o):
        return self.num2Class[(np.where(o==o.max()))[0][0]]
    
class BinaryClassCoder(ClassCoder):
    
    def _encode_class(self,o):
        return self.num2Class[(np.where(o==o.max()))[0][0]]    
        
        
class ImageCoder:
    
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
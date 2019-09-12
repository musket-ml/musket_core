from musket_core.visualization import *
from musket_core.datasets import PredictionItem
import imageio
import sys
import numpy as np
import imgaug
from imgaug.augmentables.segmaps import SegmentationMapOnImage

@dataset_visualizer
@visualize_as_text
def default_visualizer(val:PredictionItem):
    r=str(val.x)+","+str(val.y)
    if val.prediction is not None:
        r=r+" - "+str(val.prediction)
    return r    

@dataset_visualizer
@visualize_as_image
def image_visializer(p:PredictionItem):
    cache_path=context().path
    path = cache_path + str(p.id) + ".png"
    if os.path.exists(path):
        return path    
    imageio.imwrite(path,p.x)    
    return path


@dataset_visualizer
@visualize_as_image
def image_with_mask_visializer(p:PredictionItem):
    cache_path=context().path
    path = cache_path + str(p.id) + "mask.png"
    if os.path.exists(path):
        return path
    arr=np.copy(p.x)
    res=imgaug.SegmentationMapsOnImage(p.y, p.y.shape).draw()
    for i in res:
        arr[np.mean(i,axis=2)!=0]=i[np.mean(i,axis=2)!=0]
    imageio.imwrite(path,arr)    
    return path

@dataset_visualizer
@visualize_as_image
def image_with_mask_and_prediction_visializer(p:PredictionItem):
    cache_path=context().path
    path = cache_path + str(p.id) + "mask.png"
    if os.path.exists(path):
        return path
    arr=np.copy(p.x)
    res=imgaug.SegmentationMapsOnImage(p.y, p.y.shape).draw()
    for i in res:
        arr[np.mean(i,axis=2)!=0]=i[np.mean(i,axis=2)!=0]
        
    res=imgaug.SegmentationMapsOnImage(p.prediction>0.5, p.y.shape).draw()
    arr1=np.copy(p.x)
    for i in res:
        arr1[np.mean(i,axis=2)!=0]=i[np.mean(i,axis=2)!=0]    
    
    imgs_comb = np.hstack([arr,arr1])
    imageio.imwrite(path,imgs_comb)    
    return path
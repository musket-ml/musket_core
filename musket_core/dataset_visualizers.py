from musket_core.visualization import *
from musket_core.datasets import PredictionItem
import imageio
import sys
import numpy as np
import imgaug

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
    res=imgaug.SegmentationMapOnImage(p.y, p.y.shape).draw_on_image(arr)    
    imageio.imwrite(path,res)    
    return path

@dataset_visualizer
@visualize_as_image
def image_with_mask_and_prediction_visializer(p:PredictionItem):
    cache_path=context().path
    path = cache_path + str(p.id) + "mask.png"
    if os.path.exists(path):
        return path
    arr=np.copy(p.x)
    res=imgaug.SegmentationMapOnImage(p.y, p.y.shape).draw_on_image(arr)    
    
    arr1=np.copy(p.x)
    res1=imgaug.SegmentationMapOnImage(p.prediction, p.prediction.shape).draw_on_image(arr1)
    
    imgs_comb = np.hstack([res,res1])
    imageio.imwrite(path,imgs_comb)    
    return path
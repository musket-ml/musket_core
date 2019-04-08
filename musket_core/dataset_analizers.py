from musket_core import visualization,datasets
from musket_core import dataset_analis
import  numpy as np

@visualization.dataset_analizer
def dataset_balance(y):
    s=(y>0.5).sum()
    if s>0:
        return "Positive samples"
    return "Negative samples"


@visualization.dataset_analizer
def onehot_analizer(y):
    r=np.where(y>0.5)
    if len(r[0])>0:
        return int(r[0][0])
    return "Negative sample"


@visualization.dataset_analizer
def class_num_analizer(y):
    if isinstance(y, np.ndarray):
        return int(y[0])
    return int(y)

@visualization.dataset_analizer
def input_shape(p:datasets.PredictionItem):
    return str(dataset_analis.get_shape(p.x))

@visualization.dataset_analizer
def multi_class_balance(y):
    if isinstance(y, np.ndarray):
        return int(y[0])
    return int(y)

@visualization.dataset_filter
def contains_string(y:datasets.PredictionItem,filter:str):
    val=y.x
    return filter in str(val)

@visualization.dataset_filter
def custom_python(y:datasets.PredictionItem,filter:str):
    val=y.x
    return filter in str(val)


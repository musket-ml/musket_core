from musket_core import visualization
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
def multi_class_balance(y):
    if isinstance(y, np.ndarray):
        return int(y[0])
    return int(y)
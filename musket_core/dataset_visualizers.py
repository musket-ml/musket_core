from musket_core.visualization import *
from musket_core.datasets import PredictionItem

@dataset_visualizer
@visualize_as_text
def default_visualizer(val:PredictionItem):
    r=str(val.x)+","+str(val.y)
    if val.prediction is not None:
        r=r+" - "+str(val.prediction)
    return r    
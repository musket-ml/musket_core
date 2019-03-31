from musket_core.visualization import *
from musket_core.datasets import PredictionItem

@dataset_visualizer
@require_original
@visualize_as_text
def default_visualizer(val:PredictionItem):
    return str(val.x)+","+str(val.y)
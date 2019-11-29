# Instance Segmentation Training Pipeline

Instance Segmentation Pipeline was developed in order to enable using the [MMDetection](https://github.com/open-mmlab/mmdetection)
framework by means of Musket ML.

*This package is not yet completely ready for production use*

## Installation

```
pip install git+https://github.com/musket_ml/instance_segmentation_pipeline
```
*Note: this package requires python 3.6*

This package is not a part of [Musket ML](https://musket-ml.com/) framework yet,


## Launching

## Launching experiments

`fit.py` script is designed to launch experiment training.

In order to run the experiment or a number of experiments,   

A typical command line may look like this:

`musket fit --project "path/to/project" --name "experiment_name" --num_gpus=1 --gpus_per_net=1 --num_workers=1 --cache "path/to/cache/folder"`

[--project](reference.md#fitpy-project) points to the root of the [project](#project-structure)

[--name](reference.md#fitpy-name) is the name of the project sub-folder containing experiment yaml file.

[--num_gpus](reference.md#fitpy-num_gpus) sets number of GPUs to use during experiment launch.

[--gpus_per_net](reference.md#fitpy-gpus_per_net) is a maximum number of GPUs to use per single experiment.

[--num_workers](reference.md#fitpy-num_workers) sets number of workers to use.

[--cache](reference.md#fitpy-cache) points to a cache folder to store the temporary data.

Other parameters can be found in the [fit script reference](reference.md#fit-script-arguments)

### Launching tasks

`task.py` script is designed to launch experiment training.
 
Tasks must be defined in the project python scope and marked by an 
annotation like this:

```python
from musket_core import tasks, model
@tasks.task
def measure2(m: model.ConnectedModel):
    return result
```

In order to run the experiment or a number of experiments,   

A typical command line may look like this:

`python -m musket_core.task --project "path/to/project" --name "experiment_name" --task "task_name" --num_gpus=1 --gpus_per_net=1 --num_workers=1 --cache "path/to/cache/folder"`

[--project](reference.md#taskpy-project) points to the root of the [project](#project-structure)

[--name](reference.md#taskpy-name) is the name of the project sub-folder containing experiment yaml file.

[--task](reference.md#taskpy-name) is the name of the task function.

[--num_gpus](reference.md#taskpy-num_gpus) sets number of GPUs to use during experiment launch.

[--gpus_per_net](reference.md#taskpy-gpus_per_net) is a maximum number of GPUs to use per single experiment.

[--num_workers](reference.md#taskpy-num_workers) sets number of workers to use.

[--cache](reference.md#taskpy-cache) points to a cache folder to store the temporary data.

Other parameters can be found in the [task script reference](reference.md#task-script-arguments)

### Launching project analysis

`analize.py` script is designed to launch project-scope analysis.

Note that only experiments, which training is already finished will be covered.

`musket analize --inputFolder "path/to/project"`

[--inputFolder](reference.md#analyzepy-inputfolder) points to a folder to search for finished experiments in. Typically, project root.

Other parameters can be found in the [analyze script reference](reference.md#analyze-script-arguments)

## Usage guide

### Dataset format

The pipline must be used with datasets which return prediction items of certain structure.
Suppose that our prediction item represents a training example with some image and `N` objects on it.

* `x` must contain image data represented by a numpy array of shape `(height, width, 3)`.
* `y` must be a tuple `(labels, bboxes, masks)` where
    * `labels` is a length `N` one dimansional numpy array of integers which contains object labels. Note that zero label is reserved for background. 
    * `bboxes` is a float array of shape `(N, 4)` which contains object bounding boxes. Note that bounding box coordinates order must be `[minY, minX, maxY, maxX]`.
    * `masks` is an integer or boolean numpy array of shape `(N, height, width)` which contains object masks.
    Note that integer mask should contain `1` for object pixels and `0` for background.

### Training a model

Let's start from the absolutely minimalistic example. Let's say that you have two folders, one of them contains
jpeg images, and another one - png files with segmentation masks for these images. And you need to train a neural network
that will do segmentation for you. In this extremely simple setup all that you need is to type following 5
lines of python code:
```python
from musket_core import generic
from get_some_gataset import getDataset

ds = getDataset #some dataset which has the required format
cfg = generic.parse("config.yaml")
cfg.fit(ds)
```

Looks simple, but there is a `config.yaml` file in the code, and probably it is the place where everything actually happens.

```yaml
#%Musket MMDetection 1.0
classes: 46
shape: [800, 1333]

imagesPerGpu: 1
folds_count: 1
testSplit: 0.1

stages:
  - epochs: 3

dataset:
  getTrain2:

configPath: '../../data/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e-1.py'
weightsPath: '../../data/checkpoints/Hybrid-Task-Cascade-(HTC)/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c-nohead.pth'
```

#### General train properties

The following property is required to set:

[configPath](reference.md#configpath) path to MMDetection config.

The following ones are optional, but commonly used:

[classes](reference.md#classes) sets the number of classes that should be used.

[shape](reference.md#shape) set the desired shape of the input picture and mask, in the form heigth, width, number of channels. Input will be resized to fit.

[weightsPath](reference.md#weightspath) path to initial weights of the model.

[resetHeads](reference.md#resetheads) whether to refuse adopting pretrained weights for mask head and bounding box head.

#### Image and Mask Augmentations

Framework uses awesome [imgaug](https://github.com/aleju/imgaug) library for augmentation, so you only need to configure your augmentation process in declarative way like in the following example:
 
```yaml
augmentation:  
  Fliplr: 0.5
  Flipud: 0.5
  Affine:
    scale: [0.8, 1.5] #random scalings
    translate_percent:
      x: [-0.2,0.2] #random shifts
      y: [-0.2,0.2]
    rotate: [-16, 16] #random rotations on -16,16 degrees
    shear: [-16, 16] #random shears on -16,16 degrees
```
[augmentation](reference.md#augmentation) property defines [IMGAUG](https://imgaug.readthedocs.io) transformations sequence.
Each object is mapped on [IMGAUG](https://imgaug.readthedocs.io) transformer by name, parameters are mapped too.

In this example, `Fliplr` and `Flipud` keys are automatically mapped on [Flip agugmenters](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_flip.html),
their `0.5` parameter is mapped on the first `p` parameter of the augmenter.
Named parameters are also mapped, in example `scale` key of `Affine` is mapped on `scale` parameter of [Affine augmenter](https://imgaug.readthedocs.io/en/latest/source/augmenters.html?highlight=affine#affine).

One interesting augementation option when doing background removal task is replacing backgrounds with random 
images. We support this with `BackgroundReplacer` augmenter:

```yaml
augmentation:
  BackgroundReplacer:
    path: ./bg #path to folder with backgrounds
    rate: 0.5 #fraction of original backgrounds to preserve

```
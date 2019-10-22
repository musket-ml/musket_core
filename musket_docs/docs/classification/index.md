# Classification training pipeline

Classification pipeline is a declarative pipeline for single and multi output image classification tasks.
It can be also used for regression tasks

## Reasons to use Classification Pipeline

Classification Pipeline was developed with a focus of enabling to make fast and 
simply-declared experiments, which can be easily stored, 
reproduced and compared to each other.

Classification Pipeline has a lot of common parts with [Generic pipeline](../generic/index.md), but it is easier to define an architecture of the network.
Also there are a number of classification-specific features.

The pipeline provides the following features:

* Allows to describe experiments in a compact and expressive way
* Provides a way to store and compare experiments in order to methodically find the best deap learning solution
* Easy to share experiments and their results to work in a team
* Experiment configurations are separated from model definitions
* It is easy to configure network architecture
* Provides great flexibility and extensibility via support of custom substances
* Common blocks like an architecture, callbacks, model metrics, predictions vizualizers and others should be written once and be a part of a common library

## Installation

```
pip install classification_pipeline
```
*Note: this package requires python 3.6*

## Launching

### Launching experiments

`fit.py` script is designed to launch experiment training.
 It is located in the `musket_core` root folder.

Working directory *must* point to the `musket_core` root folder.

In order to run the experiment or a number of experiments,   

A typical command line may look like this:

`python ./fit.py --project "path/to/project" --name "experiment_name" --num_gpus=1 --gpus_per_net=1 --num_workers=1 --cache "path/to/cache/folder"`

[--project](reference.md#fitpy-project) points to the root of the [project](#project-structure)

[--name](reference.md#fitpy-name) is the name of the project sub-folder containing experiment yaml file.

[--num_gpus](reference.md#fitpy-num_gpus) sets number of GPUs to use during experiment launch.

[--gpus_per_net](reference.md#fitpy-gpus_per_net) is a maximum number of GPUs to use per single experiment.

[--num_workers](reference.md#fitpy-num_workers) sets number of workers to use.

[--cache](reference.md#fitpy-cache) points to a cache folder to store the temporary data.

Other parameters can be found in the [fit script reference](reference.md#fit-script-arguments)

### Launching tasks

`task.py` script is designed to launch experiment training.
 It is located in the `musket_core` root folder.
 
Tasks must be defined in the project python scope and marked by an 
annotation like this:

```python
from musket_core import tasks, model
@tasks.task
def measure2(m: model.ConnectedModel):
    return result
```

Working directory *must* point to the `musket_core` root folder.

In order to run the experiment or a number of experiments,   

A typical command line may look like this:

`python ./task.py --project "path/to/project" --name "experiment_name" --task "task_name" --num_gpus=1 --gpus_per_net=1 --num_workers=1 --cache "path/to/cache/folder"`

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
It is located in the `musket_core` root folder.

Note that only experiments, which training is already finished will be covered.

`python ./analize.py --inputFolder "path/to/project"`

[--inputFolder](reference.md#analyzepy-inputfolder) points to a folder to search for finished experiments in. Typically, project root.

Other parameters can be found in the [analyze script reference](reference.md#analyze-script-arguments)


## Usage guide

### Training a model 

Let's start from a simple example of classification. Suppose, your data are structured as follows: a .cvs file with images ids and their labels and a folder with all these images. For training a neural network to classify these images all you need are few lines of python code:

```python
import musket_core
from classification_pipeline import classification
class ProteinDataGenerator:

    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        X,y = self.__load_image(self.paths[idx]),self.labels[idx]
        return PredictionItem(self.paths[idx],X, y)

    def __load_image(self, path):
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        

        im = np.stack((
            np.array(R),
            np.array(G),
            np.array(B),
        ), -1)
        return im
dataset = ProteinDataGenerator(paths,labels)
cfg = classification.parse("config.yaml")
cfg.fit(dataset)
```

Looks simple, but there is a `config.yaml` file in the code, and probably it is the place where everything actually happens.

```yaml
architecture: DenseNet201 #pre-trained model we are going to use
pooling: avg
augmentation: #define some minimal augmentations on images
 Fliplr: 0.5
 Flipud: 0.5
classes: 28 #define the number of classes
activation: sigmoid #as we have multilabel classification, the activation for last layer is sigmoid
weights: imagenet #we would like to start from network pretrained on imagenet dataset
shape: [224, 224, 3] #our desired input image size, everything will be resized to fit
testSplit: 0.4
optimizer: Adam #Adam optimizer is a good default choice
batch: 16 #our batch size will be 16
lr: 0.005
copyWeights: true
metrics: #we would like to track some metrics
  - binary_accuracy
  - macro_f1
primary_metric: val_binary_accuracy #the most interesting metric is val_binary_accuracy
primary_metric_mode: max
callbacks: #configure some minimal callbacks
  EarlyStopping:
    patience: 3
    monitor: val_macro_f1
    mode: max
    verbose: 1
  ReduceLROnPlateau:
    patience: 2
    factor: 0.3
    monitor: val_binary_accuracy
    mode: max
    cooldown: 1
    verbose: 1
loss: binary_crossentropy #we use binary_crossentropy loss
stages:
  - epochs: 10 #let's go for 100 epochs
```

So as you see, we have decomposed our task in two parts, *code that actually trains the model* and *experiment configuration*,
which determines the model and how it should be trained from the set of predefined building blocks.

Moreover, the whole fitting and prediction process can be launched with built-in script, 
the only really required python code is dataset definition to let the system know, which data to load.

What does this code actually do behind the scenes?

-  it splits your data into 5 folds, and trains one model per fold;
-  it takes care of model checkpointing, generates example image/label tuples, collects training metrics. All this data will
   be stored in the folders just near your `config.yaml`;
-  All your folds are initialized from fixed default seed, so different experiments will use exactly the same train/validation splits.

Also, datasets can be specified directly in your config file in more generic way, see examples ds_1, ds_2, ds_3 in "segmentation_training_pipeline/examples/people" folder. In this case you can just call cfg.fit() without providing dataset programmatically.

Lets discover what's going on in more details:

#### General train properties

Lets take our standard example and check the following set of instructions:

```yaml
testSplit: 0.4
optimizer: Adam #Adam optimizer is a good default choice
batch: 16 #Our batch size will be 16
metrics: #We would like to track some metrics
  - binary_accuracy 
  - iou
primary_metric: val_binary_accuracy #and the most interesting metric is val_binary_accuracy
loss: binary_crossentropy #We use simple binary_crossentropy loss
```

[testSplit](reference.md#testsplit) Splits the train set into two parts, using one part for train and leaving the other untouched for a later testing.
The split is shuffled. 

[optimizer](reference.md#optimizer) sets the optimizer.

[batch](reference.md#batch) sets the training batch size.

[metrics](reference.md#metrics) sets the metrics to track during the training process. Metric calculation results will be printed in the console and to `metrics` folder of the experiment.

[primary_metric](reference.md#primary_metric) Metric to track during the training process. Metric calculation results will be printed in the console and to `metrics` folder of the experiment.
Besides tracking, this metric will be also used by default for metric-related activity, in example, for decision regarding which epoch results are better.

[loss](reference.md#loss) sets the loss function. if your network has multiple outputs, you also may pass a list of loss functions (one per output) 

Framework supports composing loss as a weighted sum of predefined loss functions. For example, following construction
```yaml
loss: binary_crossentropy+0.1*dice_loss
```
will result in loss function which is composed from `binary_crossentropy` and `dice_loss` functions.

There are many more properties to check in [Reference of root properties](reference.md#pipeline-root-properties)

#### Defining architecture

Lets take a look at the following part of our example:

```yaml
architecture: DenseNet201 #let's select  architecture that we would like to use
classes: 28 #we have just one class (mask or no mask)
activation: sigmoid #one class means that our last layer should use sigmoid activation
weights: imagenet
shape: [224, 224, 3] #This is our desired input image and mask size, everything will be resized to fit.
```

The following three properties are required to set:

[architecture](reference.md#architecture) This property configures architecture that should be used. `net`, `Linknet`, `PSP`, `FPN` and more are supported.

[classes](reference.md#classes) sets the number of classes that should be used. 

The following ones are optional, but commonly used:

[activation](reference.md#activation) sets activation function that should be used in last layer.

[shape](reference.md#shape) set the desired shape of the input picture and mask, in the form heigth, width, number of channels. Input will be resized to fit.

[weights](reference.md#weights) configures initial weights of the encoder.


#### Image Augmentations

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

#### Freezing and Unfreezing encoder

Freezing encoder is often used with transfer learning. If you want to start with frozen encoder just add

```yaml
freeze_encoder: true
stages:
  - epochs: 10 #Let's go for 10 epochs with frozen encoder
  
  - epochs: 100 #Now let's go for 100 epochs with trainable encoder
    unfreeze_encoder: true  
```

in your experiments configuration, then on some stage configuration just add

```yaml
unfreeze_encoder: true
```
to stage settings.

Both [freeze_encoder](reference.md#freeze_encoder) and [unfreeze_encoder](reference.md#unfreeze_encoder)
can be put into the root section and inside the stage.

*Note: This option is not supported for DeeplabV3 architecture.*

#### Custom datasets

You can declare your own dataset class as in this example:

```python
from musket_core.datasets import PredictionItem
import os
import imageio
import pandas as pd
import numpy as np
import cv2

class ClassificationDS:

    def __init__(self,imgPath):
        self.species = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
           'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
           'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
        self.data = []
        self.targets = []
        self.ids = []
        for s_id, s in enumerate(self.species):
            s_folder = os.path.join(imgPath,s)
            for file in os.listdir(s_folder):
                self.data.append(os.path.join(s_folder, file))
                self.targets.append(s_id)
                self.ids.append(file)
        
    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        item_file = self.data[item]
        target = self.targets[item]
        t = np.zeros(len(self.species))
        t[target] = 1.0
        image = self.read_image(item_file, (224,224))
        return PredictionItem(self.ids[item], image, t)
    
    def read_image(self, filepath, target_size=None):
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        img = cv2.resize(img.copy(), target_size, interpolation = cv2.INTER_AREA)
        return img

def getTrain()->datasets.DataSet:
    return ClassificationDS("images/")    
```   

Now, if this python code sits somewhere in python files located in `modules` folder of the project, and that file is referred by [imports](reference.md#imports) instruction, following YAML can refer it:
```yaml
dataset:
  getTrain: []
```

[dataset](reference.md#dataset) sets the main training dataset.

[datasets](reference.md#datasets) sets up a list of available data sets to be referred by other entities.


### Multi output classification

Sometimes you need to create network that performs several classification tasks at the same moment, in this situation
you need to declare `classes` , `activation` and `loss` as the lists of class counts, activation functions and losses
like in the following snippet:

```yaml
classes: [ 4, 4 ] #define the number of classes
activation: [sigmoid,sigmoid]
loss: 
  - binary_crossentropy
  - binary_crossentropy
primary_metric: val_loss #the most interesting metric is val_binary_accuracy
primary_metric_mode: min  
```

it is also very likely that you need to change primary metric to `val_loss`

####  preparing dataset for multi output classification:

When you use multi output classification your prediction item `y` should be a list of numpy arrays, like in
the following sample:

```python
return PredictionItem(self.ids[item], image, [t0,t1,t2])
````

length of this list should be equal to the number of network outputs



#### Multistage training

Sometimes you need to split your training into several stages. You can easily do it by adding several stage entries
in your experiment configuration file.

[stages](reference.md#stages) instruction allows to set up stages of the train process, where for each stage it is possible to set some specific training options like the number of epochs, learning rate, loss, callbacks, etc.
Full list of stage properties can be found [here](reference.md#stage-properties).

```yaml
stages:
  - epochs: 100 #Let's go for 100 epochs
  - epochs: 100 #Let's go for 100 epochs
  - epochs: 100 #Let's go for 100 epochs
```

```yaml
stages:
  - epochs: 6 #Train for 6 epochs
    negatives: none #do not include negative examples in your training set 
    validation_negatives: real #validation should contain all negative examples    

  - lr: 0.0001 #let's use different starting learning rate
    epochs: 6
    negatives: real
    validation_negatives: real

  - loss: lovasz_loss #let's override loss function
    lr: 0.00001
    epochs: 6
    initial_weights: ./fpn-resnext2/weights/best-0.1.weights #let's load weights from this file    
```       

#### Balancing your data

One common case is the situation when part of your images does not contain any objects of interest, like in 
[Airbus ship detection challenge](https://www.kaggle.com/c/airbus-ship-detection/overview). More over your data may
be to heavily inbalanced, so you may want to rebalance it. Alternatively you may want to inject some additional
images that do not contain objects of interest to decrease amount of false positives that will be produced by the framework.
    
These scenarios are supported by [negatives](reference.md#negatives) and 
[validation_negatives](reference.md#validation_negatives) settings of training stage configuration,
these settings accept following values:

- none - exclude negative examples from the data
- real - include all negative examples 
- integer number(1 or 2 or anything), how many negative examples should be included per one positive example   

```yaml
stages:
  - epochs: 6 #Train for 6 epochs
    negatives: none #do not include negative examples in your training set 
    validation_negatives: real #validation should contain all negative examples    

  - lr: 0.0001 #let's use different starting learning rate
    epochs: 6
    negatives: real
    validation_negatives: real

  - loss: lovasz_loss #let's override loss function
    lr: 0.00001
    epochs: 6
    initial_weights: ./fpn-resnext2/weights/best-0.1.weights #let's load weights from this file    
```

if you are using this setting your dataset class must support `isPositive` method which returns true for indexes
which contain positive examples: 

```python        
    def isPositive(self, item):
        pixels=self.ddd.get_group(self.ids[item])["EncodedPixels"]
        for mask in pixels:
            if isinstance(mask, str):
                return True;
        return False
```        
#### Advanced learning rates
##### Dynamic learning rates

![Example](https://github.com/bckenstler/CLR/blob/master/images/triangularDiag.png?raw=true)

As told in [Cyclical learning rates for training neural networks](https://arxiv.org/abs/1506.01186) CLR policies can provide quicker converge for some neural network tasks and architectures. 

![Example2](https://github.com/bckenstler/CLR/raw/master/images/cifar.png)

We support them by adopting Brad Kenstler [CLR callback](https://github.com/bckenstler/CLR) for Keras.

If you want to use them, just add [CyclicLR](reference.md#cycliclr) in your experiment configuration file as shown below: 

```yaml
callbacks:
  EarlyStopping:
    patience: 40
    monitor: val_binary_accuracy
    verbose: 1
  CyclicLR:
     base_lr: 0.0001
     max_lr: 0.01
     mode: triangular2
     step_size: 300
```

There are also [ReduceLROnPlateau](reference.md#reducelronplateau) and [LRVariator](reference.md#lrvariator) options to modify learning rate on the fly.

##### LR Finder

[Estimating optimal learning rate for your model](https://arxiv.org/abs/1506.01186) is an important thing, we support this by using slightly changed 
version of [Pavel Surmenok - Keras LR Finder](https://github.com/surmenok/keras_lr_finder)

```python
cfg = classification.parse(config.yaml)
ds = SimplePNGMaskDataSet("./train","./train_mask") - ???????????????????
finder=cfg.lr_find(ds,start_lr=0.00001,end_lr=1,epochs=5)
finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
plt.show()
finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
plt.show()
```
will result in this couple of helpful images: 

![image](https://camo.githubusercontent.com/b41aeaff00fb7b214b5eb2e5c151e7e353a7263e/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a48566a5f344c57656d6a764f57762d63514f397939672e706e67)

![image](https://camo.githubusercontent.com/834996d32bbd2edf7435c5e105b53a6b447ef083/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a38376d4b715f586f6d59794a4532396c39314b3064772e706e67)

#### Training on crops

Your images can be too large to train model on them. In this case you probably want to train model on crops. All
that you need to do is to specify number of splits per axis. For example, following lines in config 

```yaml
shape: [768, 768, 3]
crops: 3
``` 
will lead to splitting each image/mask into 9 cells (3 horizontal splits and 3 vertical splits) and training model on these splits.
Augmentations will be run separately on each cell. 
[crops](reference.md#crops) property sets the number of single dimension cells.


During prediction time, your images will be split into these cells, prediction will be executed on each cell, and then results
will be assembled in single final mask. Thus the whole process of cropping will be invisible from a consumer perspective.   

### Using trained model

Okey, our model is trained, now we need to actually do image classification. Let's say, we need to run image classification on
images in the directory and store results in csv file:

```python

predictions = []
images = []

#Now let's use best model from fold 0 to do image segmentation on images from images_to_segment
preds = cfg.predict_all_to_array(dataset_test, 0, 0)
for i, item in enumerate(dataset_test):
    images.append(dataset_test.get_id(i))
    p = np.argmax(preds[i])
    predictions.append(dataset_test.get_label(p))

#Let's store results in csv
df = pd.DataFrame.from_dict({'file': images, 'species': predictions})
df.to_csv('submission.csv', index=False)
``` 

#### Ensembling predictions

And what if you want to ensemble models from several folds? Just pass a list of fold numbers to
`predict_all_to_array` like in the following example:

```python
 cfg.predict_all_to_array(dataset_test, [0,1,2,3,4], 0)
```
Another supported option is to ensemble results from extra test time augmentation (flips) by adding keyword arg `ttflips=True`.

### Custom evaluation code 

Sometimes you need to run custom evaluation code. In such case you may use: `evaluateAll` method, which provides an iterator
on the batches containing original images, training masks and predicted masks

```python
for batch in cfg.evaluateAll(ds,2):
    for i in range(len(batch.predicted_maps_aug)):
        masks = ds.get_masks(batch.data[i])
        for d in range(1,20):
            cur_seg = binary_opening(batch.predicted_maps_aug[i].arr > d/20, np.expand_dims(disk(2), -1))
            cm = rle.masks_as_images(rle.multi_rle_encode(cur_seg))
            pr = f2(masks, cm);
            total[d]=total[d]+pr
```

### Accessing model
You may get trained keras model by calling: ```cfg.load_model(fold, stage)```.

## Analyzing experiments results

Okey, we have done a lot of experiments and now we need to compare the results and understand what works better. This repository
contains [script](segmentation_pipeline/analize.py) which may be used to analyze folder containing sub folders
with experiment configurations and results. This script gathers all configurations, diffs them by doing structural diff, then 
for each configuration it averages metrics for all folds and  generates csv file containing metrics and parameters that
was actually changed in your experiment like in the following [example](report.csv)

This script accepts following arguments:

 - inputFolder - root folder to search for experiments configurations and results
 - output - file to store aggregated metrics
 - onlyMetric - if you specify this option all other metrics will not be written in the report file
 - sortBy - metric that should be used to sort results 

Example: 
```commandline
python analize.py --inputFolder ./experiments --output ./result.py
``` 

## What is supported?

At this moment classification pipeline supports following pre-trained models:
  - [Resnet](https://arxiv.org/abs/1512.03385)
    - ResNet18
    - ResNet34
    - ResNet50
    - ResNet101
    - ResNet152
    - ResNeXt50
    - ResNeXt101
  - [VGG](https://arxiv.org/abs/1409.1556):  
     - VGG16
     - VGG19
  - [InceptionV3](https://arxiv.org/abs/1512.00567)
  - [InceptionResNetV2](https://arxiv.org/abs/1602.07261)
  - [Xception](https://arxiv.org/abs/1610.02357)
  - [MobileNet](https://arxiv.org/pdf/1704.04861.pdf)
  - [MobileNetV2](https://arxiv.org/abs/1801.04381)
  - [DenseNet](https://arxiv.org/abs/1608.06993):
    - DenseNet121
    - DenseNet169
    - DenseNet201
  - [NasNet](https://arxiv.org/abs/1707.07012):  
    - NASNetMobile
    - NASNetLarge


Each architecture also supports some specific options, list of options is documented in [segmentation RAML library](segmentation_pipeline/schemas/segmentation.raml#L166).

Supported augmentations are documented in [augmentation RAML library](segmentation_pipeline/schemas/augmenters.raml).

Callbacks are documented in [callbacks RAML library](segmentation_pipeline/schemas/callbacks.raml).  

## Custom architectures, callbacks, metrics

Classification pipeline uses keras custom objects registry to find entities, so if you need to use
custom loss function, activation or metric all that you need to do is to register it in Keras as: 

```python
keras.utils.get_custom_objects()["my_loss"]= my_loss
```

If you want to inject new architecture, you should register it in `classification.custom_models` dictionary.

For example:
```python
classification.custom.models['MyUnet']=MyUnet 
```
where `MyUnet` is a function that accepts architecture parameters as arguments and returns an instance
of keras model.



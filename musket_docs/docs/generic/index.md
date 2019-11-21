# Generic pipeline
## Reasons to use Generic Pipeline

Generic Pipeline was developed with a focus of enabling to make fast and 
simply-declared experiments, which can be easily stored, 
reproduced and compared to each other.
  
It provides the following features:

* Allows to describe experiments in a compact and expressive way
* Provides a way to store and compare experiments in order to methodically find the best deap learning solution
* Easy to share experiments and their results to work in a team
* Experiment configurations are separated from model definitions
* Allows to define custom neural networks in a declarative style, by building it from blocks
* Provides great flexibility and extensibility via support of custom substances
* Common blocks like an architecture, callbacks, model metrics, predictions vizualizers and others should be written once and be a part of a common library

All experiments are declared in YAML dialect with lots of defaults, allowing to describe an initial experiment in several lines and then set more details if needed.

Here is a relatively complex example, most of the statements can be omitted:

```yaml
imports: [ layers, preprocessors ]
declarations:
  collapseConv:
    parameters: [ filters,size, pool]
    body:
      - conv1d: [filters,size,relu ]
      - conv1d: [filters,size,relu ]
      - batchNormalization: {}
      - collapse: pool
  net:
    - repeat(2):
      - collapseConv: [ 20, 7, 10 ]

    - cudnnlstm: [40, true ]
    - cudnnlstm: [40, true ]
    - attention: 718
    - dense: [3, sigmoid]
  preprocess:
     - rescale: 10
     - get_delta_from_average
     - cache
preprocessing: preprocess
testSplit: 0.4
architecture: net
optimizer: Adam #Adam optimizer is a good default choice
batch: 12 #Our batch size will be 16
metrics: #We would like to track some metrics
  - binary_accuracy
  - matthews_correlation
primary_metric: val_binary_accuracy #and the most interesting metric is val_binary_accuracy
callbacks: #Let's configure some minimal callbacks
  EarlyStopping:
    patience: 100
    monitor: val_binary_accuracy
    verbose: 1
  ReduceLROnPlateau:
    patience: 8
    factor: 0.5
    monitor: val_binary_accuracy
    mode: auto
    cooldown: 5
    verbose: 1
loss: binary_crossentropy #We use simple binary_crossentropy loss
stages:
  - epochs: 100 #Let's go for 100 epochs
  - epochs: 100 #Let's go for 100 epochs
  - epochs: 100 #Let's go for 100 epochs
```

## Installation

### Prerequisites

The package has many prerequisites, but some of them are 
recommended to be installed manually.

Tensorflow package of versions of 1.14 and below split into CPU and GPU ones.
Moreover, Tensorflow may be more or less compatible with the version of 
CUDA/CUDNN installed.

Here is the repository containing lots of pre-built Tensorflow 
wheels for Windows: [tensorflow-windows-wheel](https://github.com/fo40225/tensorflow-windows-wheel).
It can be used to choose the wheel depending on system architecture, 
CUDA/CUDNN version, CPU/GPU and Python version.

Read more in [Tensorflow installation guide](https://www.tensorflow.org/install/pip).

Keras has no strong dependency on Tensorflow, but in our common setup they run in pair.
We used the 2.2.4 one.

Shapely requires compilation on install on Linux/MacOS or pre-built version on Windows.
Here are the [pre-built Shapely wheels for Windows](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely). 

### Choosing your installation type

It is recommended to install to a virtual environment in order to avoid dependency version conflicts.

### Global pip installation

Install Tensorflow, Keras and Shapely as described in pre-requisites.
In example, if you have downloaded particular 
Tensorflow wheel into `C:\downloads\tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl` 
and particular Shapely wheel into `C:\downloads\Shapely-1.6.4.post1-cp36-cp36m-win_amd64.whl`, run:

```
pip install C:\downloads\tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl
pip install Keras==2.2.4
pip install C:\downloads\Shapely-1.6.4.post1-cp36-cp36m-win_amd64.whl
```

```
pip install musket_ml 
```

### Virtual environment installation (recommended)

#### virtualenv installation (recommended)

This type of installation uses virtualenv manager for 
creating your virtual environment.

Create a new virtual environment:

```
virtualenv ./musket
```

This will create `musket` folder and place a copy of your python, pip and wheel inside.

Activate the new virtual environment:

On Posix systems:
```
source ./musket/bin/activate
```
On Windows:
```
.\musket\Scripts\activate
```

Install Tensorflow, Keras and Shapely as described in pre-requisites.
In example, if you have downloaded particular 
Tensorflow wheel into `C:\downloads\tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl` 
and particular Shapely wheel into `C:\downloads\Shapely-1.6.4.post1-cp36-cp36m-win_amd64.whl`, run:

```
pip install C:\downloads\tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl
pip install Keras==2.2.4
pip install C:\downloads\Shapely-1.6.4.post1-cp36-cp36m-win_amd64.whl
```

Now install musket:
```
pip install musket_ml 
```

Experiment launches and other activity should be performed when this environment is activated.

When you are done working with musket, you can deactivate the environment by 
launching:
```
virtualenv deactivate
```

#### pipenv installation

This type of installation uses pipenv manager for 
creating your virtual environment.

Install pipenv if needed:

```
pip install --user pipenv
```

Create new environment by launching:
```
mkdir musket
cd musket
pipenv --python 3.6
```

Install Tensorflow, Keras and Shapely as described in pre-requisites.
In example, if you have downloaded particular 
Tensorflow wheel into `C:\downloads\tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl` 
and particular Shapely wheel into `C:\downloads\Shapely-1.6.4.post1-cp36-cp36m-win_amd64.whl`, run:

```
pipenv install C:\downloads\tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl
pipenv install Keras==2.2.4
pipenv install C:\downloads\Shapely-1.6.4.post1-cp36-cp36m-win_amd64.whl
```

Now install musket:
```
pipenv install musket_ml 
```

Experiment launches and other activity should be performed 
when this environment is activated or using pipenv.

So, the first approach is to activate the environment by launching
```
pipenv shell
```
while inside `musket` folder.

Or, alternativelly, prefix all experiment management commands with 

`pipenv run`,
 
in example, instead of running

`musket fit --project "D:\work\salt" --name "exp01" --num_gpus=1 --gpus_per_net=1 --num_workers=1 --cache "D:\work\salt\data\cache"`
 
 run
 
`pipenv run musket fit --project "D:\work\salt" --name "exp01" --num_gpus=1 --gpus_per_net=1 --num_workers=1 --cache "D:\work\salt\data\cache"`

### Other packages

`musket_ml` joins all mainstream pipelines that belong to Musket-ML framework.
In particular, besides `musket_core` generic pipeline, it includes 
`classification_pipeline` [classification pipeline](../classification/index.md),
`segmentation_pipeline` [segmentation pipeline](../segmentation/index.md)
 and `musket_text` [text support](../text/index.md).

To install only the generic pipeline, follow the same instructions, 
but use `musket_core` wheel instead of `musket_ml`.

 

## Project structure

Each experiment is simply a folder with YAML file inside, it is easy to store and run experiment.

Project is a folder with the following structure inside:

- **project_name**
  - **experiments**
    - **experiment1**
      - config.yaml
    - **experiment2**
      - config.yaml
      - summary.yaml
      - **metrics**
        - metrics-0.0.csv
        - metrics-1.0.csv
        - metrics-2.0.csv
        - metrics-3.0.csv
        - metrics-4.0.csv
  - **modules**
    - main.py
    - arbitrary_module.py
  - common.yaml

The only required part is `experiments` folder with at least one arbitrary-named experiment subfolder having `config.yaml` file inside.
Each experiment starts with its configuration, other files are being added by the pipeline during th training.

`common.yaml` file may be added to set instructions, which will be applied to all project experiments.

`modules` folder may be added to set python files in project scope, so custom yaml declarations can be mapped 
onto python classes and functions defined inside such files. all modules in this folder will be always executed, other modules require [imports](reference.md#imports) instruction. 

`summary.yaml` and `metrics` folders inside each experiment appear after the experiment training is executed.

There are more potential files, like intermediate results cache files etc.

## Launching

### Launching experiments

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

## General train properties

Lets take our standard example and check the following set of instructions:

```yaml
imports: [ layers, preprocessors ]
testSplit: 0.4
optimizer: Adam #Adam optimizer is a good default choice
batch: 12 #Our batch size will be 16
metrics: #We would like to track some metrics
  - binary_accuracy
  - matthews_correlation
primary_metric: val_binary_accuracy #and the most interesting metric is val_binary_accuracy
loss: binary_crossentropy #We use simple binary_crossentropy loss
```

[imports](reference.md#imports) imports python files that are not located in  `modules` folder of the project and make their properly annotated contents to be available to be referred from YAML. Files from the modules folder are imported automatically

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

## Definining networks

Lets check the next part of our example:

```yaml
declarations:
  collapseConv:
    parameters: [ filters,size, pool]
    body:
      - conv1d: [filters,size,relu ]
      - conv1d: [filters,size,relu ]
      - batchNormalization: {}
      - collapse: pool
  net:
    - repeat(2):
      - collapseConv: [ 20, 7, 10 ]

    - cudnnlstm: [40, true ]
    - cudnnlstm: [40, true ]
    - attention: 718
    - dense: [3, sigmoid]
architecture: net
```
Here, `declarations` instruction set up network blocks `collapseConv` and `net`.
`collapseConv` block defines its input parameters (those are YAML-level parameters, not actual network tensors),
and `body` defines the sub-blocks of the block.

`net` block has no parameters, so its sub-blocks come right inside the `net`.
Following are built-in layers used inside both blocks:

* [conv1d](reference.md#conv1d)
* [batchNormalization](reference.md#batchnormalization)
* [cudnnlstm](reference.md#cudnnlstm)
* [attention](reference.md#attention)
* [dense](reference.md#dense)

And data / control-flow instructions:

* [collapse](reference.md#collapse)
* [repeat](reference.md#repeat)

Also, `net` block uses `collapseConv` block by stating `collapseConv: [ 20, 7, 10 ]`, where `collapseConv` ordered parameters `[ 20, 7, 10 ]` come in YAML array.
 
[architecture](reference.md#architecture) instruction sets `net` block as the entry point for the whole experiment.

### Built-in NN layers

There are a lot of built-in NN layers, basically, we support all layers that are supported by Keras.

Here are just a few:

* [Dropout](reference.md#dropout)
* [LSTM](reference.md#lstm)
* [GlobalMaxPool1D](reference.md#globalmaxpool1d)
* [BatchNormalization](reference.md#batchnormalization)
* [Concatenate](reference.md#concatenate)
* [Conv2D](reference.md#conv2d)
* [Dense](reference.md#dense)

More can be found here: [Layer types](reference.md#layer-types)

### Control layers

[Utility layers](reference.md#utility-layers) can be used to set control and data flow inside their bodies. Here are some examples:

#### Simple Data Flow constructions

```yaml
  inceptionBlock:
    parameters: [channels]
    with:
      padding: same
    body:
      - split-concatenate:
        - Conv2D: [channels,1]
        - seq:
          - Conv2D: [channels*3,1]
          - Conv2D: [channels,3]
        - seq:
            - Conv2D: [channels*4,1]
            - Conv2D: [channels,1]
        - seq:
            - Conv2D: [channels,2]
            - Conv2D: [channels,1]            
```            

#### Repeat and With

```yaml
declarations:
  convBlock:
    parameters: [channels]
    with:
      padding: same
    body:
      - repeat(5):
        - Conv2D: [channels*_,1]
  net:
      - convBlock: [120]
```

#### Conditional layers

```yaml
declarations:
  c2d:
    parameters: [size, pool,mp]
    body:
      - Conv1D: [100,size,relu]
      - Conv1D: [100,size,relu]
      - Conv1D: [100,size,relu]
      - if(mp):
          MaxPool1D: pool
  net:
      - c2d: [4,4,False]
      - c2d: [4,4,True]
      - Dense: [4, sigmoid]
```

#### Shared Weights

```yaml
#Basic example with sequencial model
declarations:
  convBlock:
    parameters: [channels]
    shared: true
    with:
      padding: same
    body:
      - Conv2D: [channels,1]
      - Conv2D: [channels,1]
  net:
      - convBlock: [3] #weights of convBlock will be shared between invocations
      - convBlock: [3] #weights of convBlock will be shared between invocations
```

#### Wrapper layers

```yaml
  net:
    #- gaussianNoise: 0.0001

    #- collapseConv: [ 20, 7, 10 ]
    #- collapseConv: [ 20, 7, 10 ]
    - bidirectional:
        - cudnnlstm: [30, true ]
    - bidirectional:
        - cudnnlstm: [50, true ]
    - attention: 200
    - dense: [64, relu]
    - dense: [3, sigmoid]
```    

#### Manually controlling data flow
```yaml
  net:
    inputs: [i1,i2]
    outputs: [d1,d2]
    body:
      - c2d:
          args: [4,4]
          name: o1
          inputs: i1
      - c2d:
          args: [4,4]
          name: o2
          inputs: i2
      - dense:
          units: 4
          activation: sigmoid
          inputs: o1
          name: d1
      - dense:
          units: 4
          activation: sigmoid
          inputs: o2
          name: d2
```

Full list can be found [here](reference.md#utility-layers)

## Datasets

Datasets allow to define the ways to load data for this particular project.
As this pipeline is designed to support an arbitrary data, the only way to add dataset is to put in some custom python code and then refer it from YAML:

```python
class DischargeData(datasets.DataSet):

    def __init__(self,ids,normalize=True, flatten=False):
        self.normalize=normalize
        self.flatten = flatten
        self.cache={}
        self.ids=list(set(list(ids)))

    def __getitem__(self, item):
        item=self.ids[item]
        if item in self.cache:
            return self.cache[item]
        ps= PredictionItem(item,getX(item,self.normalize),getY(item,self.flatten))
        #self.cache[item]=ps
        return ps

    def __len__(self):
        return len(self.ids)

def getTrain(normalize=True,flatten=False)->datasets.DataSet:
    return DischargeData(ids,normalize,flatten)

def getTest(normalize=True,flatten=False)->datasets.DataSet:
    return DischargeData(test_ids,normalize,flatten)    
```

Now, if this python code sits somewhere in python files located in `modules` folder of the project, and that file is referred by [imports](reference.md#imports) instruction, following YAML can refer it:
```yaml
dataset:
  getTrain: [false,false]
datasets:
  test:
    getTest: [false,false]
```

[dataset](reference.md#dataset) sets the main training dataset.

[datasets](reference.md#datasets) sets up a list of available data sets to be referred by other entities.
                                  
## Callbacks

Lets check the following block from out main example:

```yaml
callbacks: #Let's configure some minimal callbacks
  EarlyStopping:
    patience: 100
    monitor: val_binary_accuracy
    verbose: 1
  ReduceLROnPlateau:
    patience: 8
    factor: 0.5
    monitor: val_binary_accuracy
    mode: auto
    cooldown: 5
    verbose: 1
```

We set up two callback, which are being invoked during the training time: 
[EarlyStopping](reference.md#earlystopping) that monitors metrics and stops training if results doesnt get better, and `val_binary_accuracy` and [ReduceLROnPlateau](reference.md#reducelronplateau), which reduces learning rate for the same reason.

The list of callbacks can be found [here](reference.md#callbacks)

## Stages

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

## Balancing your data

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

## Advanced learning rates
### Dynamic learning rates

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

### LR Finder

[Estimating optimal learning rate for your model](https://arxiv.org/abs/1506.01186) is an important thing, we support this by using slightly changed 
version of [Pavel Surmenok - Keras LR Finder](https://github.com/surmenok/keras_lr_finder)

```python
cfg= segmentation.parse(people-1.yaml)
ds=SimplePNGMaskDataSet("./train","./train_mask")
finder=cfg.lr_find(ds,start_lr=0.00001,end_lr=1,epochs=5)
finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
plt.show()
finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
plt.show()
```
will result in this couple of helpful images: 

![image](https://camo.githubusercontent.com/b41aeaff00fb7b214b5eb2e5c151e7e353a7263e/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a48566a5f344c57656d6a764f57762d63514f397939672e706e67)

![image](https://camo.githubusercontent.com/834996d32bbd2edf7435c5e105b53a6b447ef083/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a38376d4b715f586f6d59794a4532396c39314b3064772e706e67)

## [Preprocessors](reference.md#preprocessors)

[Preprocessors](reference.md#preprocessors) are the custom python functions that transform dataset. 

Such functions should be defined in python files that are in a project scope (`modules`) folder and imported.
Preprocessing functions should be also marked with `@preprocessing.dataset_preprocessor` annotation.

[preprocess](reference.md#preprocess) instruction then can be used to chain preprocessors as needed for this particular experiment, and even cache the result on disk to be reused between experiments.


```yaml
preprocess:
     - rescale: 10
     - get_delta_from_average
     - disk-cache
```

```python
import numpy as np
from musket_core import preprocessing

def moving_average(input, n=1000) :
    ret = np.cumsum(input, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    ret[0:n] = ret[-n:]
    return ret / n

@preprocessing.dataset_preprocessor
def get_delta_from_average(input):
    m = moving_average(input[:, :])
    m1 = moving_average(input[:, :],100)
    #m2 = moving_average(input[:, :], 10000)
    d = input[:, :] - m
    d1 = input[:, :] - m1
    #d2 = input[:, :] - m2

    input=input/input.max()
    d1 = d1 / d1.max()
   # d2 = d2 / d2.max()
    d = d / d.max()
    return np.concatenate([d,d1,input])

@preprocessing.dataset_preprocessor
def rescale(input,size):
    mean=np.mean(np.reshape(input, (input.shape[0] // size ,size, 3)), axis=1)
    max=np.max(np.reshape(input, (input.shape[0] // size, size, 3)), axis=1)
    min = np.min(np.reshape(input, (input.shape[0] // size, size, 3)), axis=1)
    return np.concatenate([mean,max,min])
```

## How to check training results

In experiment folder `metrics` subfolder contain a CSV report file for each fold and stage.

`summary.yaml` file in the experiment folder contain the statistics for the whole experiment.
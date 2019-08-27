# Generic pipeline
## Reasons to use Generic Pipeline

TODO: add more text from a general promo here

Generic Pipeline was developed with a focus of enabling to make fast and simply-declared experiments, which can be easily stored, reproduced and compared to each other.
  
It provides the following features:

* Allows to describe experiments in a compact and expressive way
* Provides a way to store and compare experiments in order to methodically find the best deap learning solution
* Easy to share experiments and their results to work in a team
* Allows to define custom neural networks in a declarative style, by building it from blocks
* Provides great flexibility and extensibility via support of custom substances

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

TODO: make sure this actually works.
```pip install generic_pipeline```

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
onto python classes and functions defined inside such files. `main.py` will be always executed, other files require [imports](reference.md#imports) instruction. 

`summary.yaml` and `metrics` folders inside each experiment appear after the experiment training is executed.

There are more potential files, like intermediate results cache files etc.

## Launching

TODO

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

[stages](reference.md#stages) instruction allows to set up stages of the train process, where for each stage it is possible to set some specific training options like the number of epochs, learning rate, loss, callbacks, etc.
Full list of stage properties can be found [here](reference.md#stage-properties).


```yaml
stages:
  - epochs: 100 #Let's go for 100 epochs
  - epochs: 100 #Let's go for 100 epochs
  - epochs: 100 #Let's go for 100 epochs
```

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
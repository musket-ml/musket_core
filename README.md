# dl_pipeline_core
Core Functionality of DL Pipelines

### Tensorboard

If you want to add tensorboard - just add callback configuration into your yaml experiment configuration file 

```yaml
callbacks:
  TensorBoard:
    log_dir: './logs'
    batch_size: 32
    write_graph: True
    update_freq: batch
```    
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
    #- gaussianNoise: 0.0001
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
# Network Definitions



## Defining simple network

```yaml
   net:
      - conv1D: [100,4,relu]
      - conv1D: [100,4,relu]
      - conv1D: [100,4,relu]
      - maxPool1D: 8
      - dense: [2,softmax]      
```

alternatively:


```yaml
   conv1D: 
      filters:100
      kernel_size: 4
      activation: relu      
```

## Reusable modules
```yaml
#Basic example with sequencial model
declarations:
  c2d:
    parameters: [size, pool]
    body:
      - Conv1D: [100,size,relu]
      - Conv1D: [100,size,relu]
      - Conv1D: [100,size,relu]
      - MaxPool1D: pool
  net:
      - c2d: [4,4]
      - c2d: [4,4]
      - Dense: [4, sigmoid]
```

## Controlling Data Flow


### Simple Data Flow constructions

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

### Repeat and With

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

### Conditional layers

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

### Shared Weights

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

### Wrapper layers

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

### Manually controlling data flow
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
## Plugin external definitions


# Data Preprocessing

```yaml
preprocess:
     - rescale: 10
     - get_delta_from_average
     - cache
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



## Gradient Boosting
Example config: musket_core/examples/gb_multiclass.yaml

## Hyperparameters Search
gradient boosting based examle:
```
#%Musket GradientBoosting !
imports: [data]

declarations:
  net:
    - gradientboosting:
        output_dim: 1
        num_leaves: x

architecture: net

validationSplit: 0.3

folds_count: 1

metrics:
  - binary_accuracy
  - binary_crossentropy
  - matthews_correlation
  - macro_f1

primary_metric: macro_f1
primary_metric_mode: max

loss: regression

max_evals: 3

hyperparameters:
  x:
    enum: [5, 6, 7]
    type: int

stages:
  - epochs: 10
dataset:
  data: []
```

possible declarations:
```
max_evals: 3

hyperparameters:
  x:
    enum: [5, 6, 7]
    type: int
```
x param will take value 5, 6, 7 (or 5.0, 6.0, 7.0 if type is 'float'), values will be taken 3 times. Uniform distribution will be used. No values will be chosen twise if size of list is less then 'max_evals'.  if 'max_evals' is equal size of list, then all values from list will be taken.

```
max_evals: 3

hyperparameters:
  x:
    range: [1, 5]
    type: int
```
x param will take random values from (1, 2, 3, 4, 5), values will be taken 3 times. Uniform distribution will be used. No values will be chosen twise if size of list is less then 'max_evals'.  if 'max_evals' is equal size of list, then all values from list will be taken.

```
max_evals: 3

hyperparameters:
  x:
    range: [1, 5]
    type: float
```

x param will take 3 random values from 1.0 to 5.0 inclusively. Uniform distribution will be used.
```
max_evals: 3

hyperparameters:
  x: [1, 5]
```
this is shortcut for same as above

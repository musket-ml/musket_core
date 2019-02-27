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

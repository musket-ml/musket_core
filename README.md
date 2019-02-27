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

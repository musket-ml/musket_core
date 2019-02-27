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
```
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

### Manually controlling data flow

## Plugin external definitions


# Data Preprocessing

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

## Reusable modules

## Controlling Data Flow

### Simple Data Flow constructions

### Manually controlling data flow

## Plugin external definitions


# Data Preprocessing

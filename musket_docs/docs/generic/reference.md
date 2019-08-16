# Generic pipeline reference
# Pipeline root properties
## activation
TODO: does it have any use in the root of the file?
## aggregation_metric

**type**: ``string`` 

Metric to calculate against the combination of all stages and report in `allStages` section of summary.yaml file after all experiment instances are finished.

Uses metric name detection mechanism to search for the built-in metric or for a custom function with the same name across project modules.

Metric name may have `val_` prefix or `_holdout` postfix to indicate calculation against validation or holdout, respectively. 

Example:
```yaml
aggregation_metric: matthews_correlation_holdout

```
## architecture

**type**: ``string`` 

Name of the [declaration](#declarations) that will be used as an entry point or root of the main network. 

Example:
```yaml
declarations: 
   utilityDeclaration1:
   utilityDeclaration2:
   mainNetwork:
       - utilityDeclaration1: []
       - dense: [1,"sigmoid"]

architecture: mainNetwork

```

## augmentation

**type**: ```` 

TODO: does it have any use in the root of the file?


Example:
```yaml

```
## batch

**type**: ``integer`` 

Sets up training batch size.

Example:
```yaml
batch: 512
```
## classes

**type**: ```` 

TODO: does it have any use in generic pipeline?


Example:
```yaml

```
## callbacks

**type**: ``array of callback instances`` 

Sets up training-time callbacks. See individual [callback descriptions](#callback-types).

Example:
```yaml
callbacks:
  EarlyStopping:
    patience: 100
    monitor: val_binary_accuracy
    verbose: 1
  ReduceLROnPlateau:
    patience: 16
    factor: 0.5
    monitor: val_binary_accuracy
    mode: auto
    cooldown: 5
    verbose: 1
```
## copyWeights

**type**: ``boolean`` 

Whether to copy saved weights.

Example:
```yaml
copyWeights: true
```
## clipnorm

**type**: ```` 

Maximum clip norm of a gradient for an optimizer.

Example:
```yaml
clipnorm: 1.0
```
## clipvalue

**type**: ```` 

Clip value of a gradient for an optimizer.

Example:
```yaml
clipvalue: 0.5
```
## dataset

**type**: ``complex object`` 

Key is a name of the python function in scope, which returns training data set.
Value is an array of parameters to pass to a function.

Example:
```yaml
dataset:
  getTrain: [false,false]
```
## datasets

**type**: ``map containing complex objects`` 

Sets up a list of available data sets to be referred by other entities.

For each object, key is a name of the python function in scope, which returns training dataset.
Value is an array of parameters to pass to a function.

Example:
```yaml
datasets:
  test:
    getTest: [false,false]
```
## dataset_augmenter

**type**: ``complex object`` 

Sets up a custom augmenter function to be applied to a dataset.
Object must have a name property, whic will be used as a name of the python function in scope.
Other object properties are mapped as function arguments.
TODO: check that description and example are correct?

Example:
```yaml
dataset_augmenter:
    name: TheAugmenter
    parameter: test
```
## dropout

**type**: ```` 

TODO: does it have any use in generic pipeline root?

Example:
```yaml

```
## declarations

**type**: ```` 

Sets up network layer building blocks. 

Each declaration is an object with a key setting up declaration name
and value being a complex object containing `parameters` array listing 
this layer parameters and `body` containing an array of sub-layers or control statements,

If layer has no parameters, `parameters` property may be ommitted and `body` contents may 
come directly inside layer definition.

See [Layer types](#layer-types) for details regarding building blocks.

Example:
```yaml
declarations: 
   lstm2: 
      parameters: [count]
      body:
       - bidirectional:  
           - cuDNNLSTM: [count, true]
       - bidirectional:    
           - cuDNNLSTM: [count/2, false]
   net:
       - split-concat: 
          - word_indexes_embedding:  [ embeddings/glove.840B.300d.txt ]
          - word_indexes_embedding:  [ embeddings/paragram_300_sl999.txt ]
          - word_indexes_embedding:  [ embeddings/wiki-news-300d-1M.vec]
       - gaussianNoise: 0.05   
       - lstm2: [300]
       #- dropout: 0.5
       - dense: [1,"sigmoid"]
```
## extra_train_data

**type**: ```` 

D

Example:
```yaml

```
## folds_count

**type**: ```` 

D

Example:
```yaml

```
## freeze_encoder

**type**: ```` 

D

Example:
```yaml

```
## final_metrics

**type**: ``array of strings`` 

Metrics to calculate against every stage and report in `stages` section of summary.yaml file after all experiment instances are finished.

Uses metric name detection mechanism to search for the built-in metric or for a custom function with the same name across project modules.

Metric name may have `val_` prefix or `_holdout` postfix to indicate calculation against validation or holdout, respectively.

Example:
```yaml
final_metrics: [measure]

```
## holdout

**type**: ```` 

D

Example:
```yaml

```
## imports

**type**: ``array of strings`` 

Imports python files from `modules` folder of the project and make their properly annotated contents to be available to be referred from YAML.

Example:
```yaml
imports: [ layers, preprocessors ]

```
this will import `layers.py` and `preprocessors.py`
## inference_batch

**type**: ```` 

D

Example:
```yaml

```
## loss

**type**: ```` 

Sets the loss name.

Uses loss name detection mechanism to search for the built-in loss or for a custom function with the same name across project modules.

Example:
```yaml
loss: binary_crossentropy
```
## lr

**type**: ```` 

D

Example:
```yaml

```
## metrics

**type**: ``array of strings`` 

Array of metrics to track during the training process. Metric calculation results will be printed in the console and to `metrics` folder of the experiment.

Uses metric name detection mechanism to search for the built-in metric or for a custom function with the same name across project modules.

Metric name may have `val_` prefix or `_holdout` postfix to indicate calculation against validation or holdout, respectively.

Example:
```yaml
metrics: #We would like to track some metrics
  - binary_accuracy
  - binary_crossentropy
  - matthews_correlation

```
## num_seeds

**type**: ```` 

D

Example:
```yaml

```
## optimizer

**type**: ``string`` 

Sets the optimizer.

Example:
```yaml
optimizer: Adam
```
## primary_metric

**type**: `string` 

Metric to track during the training process. Metric calculation results will be printed in the console and to `metrics` folder of the experiment.

Besides tracking, this metric will be also used by default for metric-related activity, in example, for decision regarding which epoch results are better.

Uses metric name detection mechanism to search for the built-in metric or for a custom function with the same name across project modules.

Metric name may have `val_` prefix or `_holdout` postfix to indicate calculation against validation or holdout, respectively.

Example:
```yaml
primary_metric: val_macro_f1
```
## primary_metric_mode

**type**: ``enum: auto,min,max`` 

**default**: ``auto`` 

In case of a usage of a primary metrics calculation results across several instances (i.e. batches), this will be a mathematical operation to find a final result.

Example:
```yaml
primary_metric_mode: max
```
## preprocessing

**type**: ```` 

D

Example:
```yaml

```
## random_state

**type**: ```` 

D

Example:
```yaml

```
## stages

**type**: ```` 

D

Example:
```yaml

```
## stratified

**type**: ```` 

D

Example:
```yaml

```
## testSplit

**type**: `float 0-1` 

Splits the train set into two parts, using one part for train and leaving the other untouched for a later testing.
The split is shuffled.

Example:
```yaml
testSplit: 0.4
```
## testSplitSeed

**type**: ```` 

D

Example:
```yaml

```
## testTimeAugmentation

**type**: ```` 

D

Example:
```yaml

```
## transforms

**type**: ```` 

D

Example:
```yaml

```
## validationSplit

**type**: ```` 

D

Example:
```yaml

```

# Callback types

## EarlyStopping

Stop training when a monitored metric has stopped improving.

Properties:

* **patience** - integer, number of epochs with no improvement after which training will be stopped.
* **verbose** - 0 or 1, verbosity mode.
* **monitor** - string, name of the metric to monitor
* **mode** - auto, min or max; In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity. 

Example
```yaml
callbacks:
  EarlyStopping:
    patience: 100
    monitor: val_binary_accuracy
    verbose: 1
```

## ReduceLROnPlateau

Reduce learning rate when a metric has stopped improving.

Properties:

* **patience** - integer, number of epochs with no improvement after which training will be stopped.
* **cooldown** - integer, number of epochs to wait before resuming normal operation after lr has been reduced.
* **factor** - number, factor by which the learning rate will be reduced. new_lr = lr * factor
* **verbose** - 0 or 1, verbosity mode.
* **monitor** - string, name of the metric to monitor
* **mode** - auto, min or max; In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.

Example
```yaml
callbacks:
  ReduceLROnPlateau:
    patience: 16
    factor: 0.5
    monitor: val_binary_accuracy
    mode: auto
    cooldown: 5
    verbose: 1
```

## CyclicLR

Cycles learning rate across epochs.

Functionally, it defines the cycle amplitude (max_lr - base_lr).
The lr at any cycle is the sum of base_lr
and some scaling of the amplitude; therefore
max_lr may not actually be reached depending on
scaling function.

Properties:

* **base_lr** - number, initial learning rate which is the lower boundary in the cycle.
* **max_lr** - number, upper boundary in the cycle.
* **mode** - one of `triangular`, `triangular2` or `exp_range`; scaling function.
* **gamma** - number from 0 to 1, constant in 'exp_range' scaling function.
* **step_size** - integer > 0, number of training iterations (batches) per half cycle.

Example
```yaml
callbacks:
  CyclicLR:
    base_lr: 0.001
    max_lr: 0.006
    step_size: 2000
    mode: triangular
```

## LRVariator

Changes learning rate between two values

Properties:

* **fromVal** - initial learning rate value, defaults to the configuration LR setup.
* **toVal** - final learning value.
* **style** - one of the following:
  * **linear** - changes LR linearly between two values.
  * **const** - does not change from initial value.
  * **cos+** - `-1 * cos(2x/pi) + 1 for x in [0;1]`
  * **cos-** - `cos(2x/pi) for x in [0;1]`
  * **cos** - same as 'cos-'
  * **sin+** - `sin(2x/pi) x in [0;1]`
  * **sin-** - `-1 * sin(2x/pi) + 1 for x in [0;1]`
  * **sin** - same as 'sin+'
  * **any positive float or integer value** - x^a for x in [0;1]

TODO: examples from lr_variation_callback.py look strange,
also it is unclear how to number of steps is being set up.
Example
```yaml
```

## TensorBoard

This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of your training and test metrics, as well as activation histograms for the different layers in your model.

Properties:

* **log_dir** - string; the path of the directory where to save the log files to be parsed by TensorBoard.
* **histogram_freq** - integer; frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations.
* **batch_size** - integer; size of batch of inputs to feed to the network for histograms computation.
* **write_graph** - boolean; whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
* **write_grads** - boolean; whether to visualize gradient histograms in TensorBoard.  histogram_freq must be greater than 0.
* **write_images** - boolean; whether to write model weights to visualize as image in TensorBoard.
* **embeddings_freq** - number; frequency (in epochs) at which selected embedding layers will be saved. If set to 0, embeddings won't be computed. Data to be visualized in TensorBoard's Embedding tab must be passed as embeddings_data.
* **embeddings_layer_names** - array of strings; a list of names of layers to keep eye on. If None or empty list all the embedding layer will be watched.
* **embeddings_metadata** - a dictionary which maps layer name to a file name in which metadata for this embedding layer is saved. See the details about metadata files format. In case if the same metadata file is used for all embedding layers, string can be passed.
* **embeddings_data** -  data to be embedded at layers specified in  embeddings_layer_names. 
* **update_freq** - `epoch` or `batch` or integer; When using 'batch', writes the losses and metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 10000, the callback will write the metrics and losses to TensorBoard every 10000 samples. Note that writing too frequently to TensorBoard can slow down your training.

Example
```yaml
callbacks:
  TensorBoard:
    log_dir: './logs'
    batch_size: 32
    write_graph: True
    update_freq: batch
```

# Layer types

## Input

TODO: description

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **shape** - array of integers; input shape

Example:
```yaml

```
## GaussianNoise

Apply additive zero-centered Gaussian noise.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **stddev** - float; standard deviation of the noise distribution.

Example:
```yaml

```

## Dropout

Applies Dropout to the input.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **rate** - float; float between 0 and 1. Fraction of the input units to drop.

Example:
```yaml
declarations:
  net:
    - dropout: 0.5
```
## SpatialDropout1D

Spatial 1D version of Dropout.

This version performs the same function as Dropout, however it drops entire 1D feature maps instead of individual elements. If adjacent frames within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout1D will help promote independence between feature maps and should be used instead.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **rate** - float between 0 and 1. Fraction of the input units to drop.

Example:
```yaml

```
## LSTM

Long Short-Term Memory layer

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **units** - Positive integer, dimensionality of the output space.
* **return_sequences** - Boolean. Whether to return the last output in the output sequence, or the full sequence.
* **return_state** - Boolean. Whether to return the last state in addition to the output. The returned elements of the states list are the hidden state and the cell state, respectively.
* **stateful** - Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.

Example:
```yaml

```
## GlobalMaxPool1D

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## GlobalAveragePooling1D

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## BatchNormalization

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## Concatenate

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## Add

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## Substract

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```

## Mult

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## Max

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## Min

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## Conv1D

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## Conv2D

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## MaxPool1D

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## MaxPool2D

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## AveragePooling1D

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## CuDNNLSTM

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## Dense

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## Flatten

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## Bidirectional

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```

# Utility layers

## split

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-concat

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-concatenate

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-add

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-substract

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-mult

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-min

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-max

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-dot

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-dot-normalize

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## seq

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## input

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## cache

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## disk-cache

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-preprocessor

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-concat-preprocessor

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## seq-preprocessor

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```

## augmentation

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## pass

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## transform-concat

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## transform-add

D

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

``` 

# Stage properties

## loss
## initial_weights
## epochs
## unfreeze_encoder
## lr
## callbacks
## extra_callbacks


## Preprocessors
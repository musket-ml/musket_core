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

**type**: ``string`` 

Sets the loss name.

Uses loss name detection mechanism to search for the built-in loss or for a custom function with the same name across project modules.

Example:
```yaml
loss: binary_crossentropy
```
## lr

**type**: ``float`` 

Learning rate.

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

**type**: ``complex`` 

Preprocessors are the custom python functions that transform dataset. 

Such functions should be defined in python files that are in a project scope (`modules`) folder and imported.
Preprocessing functions should be also marked with `@preprocessing.dataset_preprocessor` annotation.

`preprocessing` instruction then can be used to chain preprocessors as needed for this particular experiment, and even cache the result on disk to be reused between experiments.

[Preprocessors](#preprocessors) contain some of the preprocessor utility instructions.

Example:
```yaml
preprocessing: 
  - binarize_target: 
  - tokenize:  
  - tokens_to_indexes:
       maxLen: 160
  - disk-cache: 
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
* **seed** - integer; integer to use as random seed

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
- __units__: Positive integer, dimensionality of the output space.
- __activation__: Activation function to use
    (see [activations](https://keras.io/activations/)).
    Default: hyperbolic tangent (`tanh`).
    If you pass `None`, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __recurrent_activation__: Activation function to use
    for the recurrent step
    (see [activations](https://keras.io/activations/)).
    Default: hard sigmoid (`hard_sigmoid`).
    If you pass `None`, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
    used for the linear transformation of the inputs.
    (see [initializers](https://keras.io/initializers/)).
- __recurrent_initializer__: Initializer for the `recurrent_kernel`
    weights matrix,
    used for the linear transformation of the recurrent state.
    (see [initializers](https://keras.io/initializers/)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](https://keras.io/initializers/)).
- __unit_forget_bias__: Boolean.
    If True, add 1 to the bias of the forget gate at initialization.
    Setting it to true will also force `bias_initializer="zeros"`.
    This is recommended in [Jozefowicz et al. (2015)](
    http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](https://keras.io/regularizers/)).
- __recurrent_regularizer__: Regularizer function applied to
    the `recurrent_kernel` weights matrix
    (see [regularizer](https://keras.io/regularizers/)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](https://keras.io/regularizers/)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](https://keras.io/regularizers/)).
- __kernel_constraint__: Constraint function applied to
    the `kernel` weights matrix
    (see [constraints](https://keras.io/constraints/)).
- __recurrent_constraint__: Constraint function applied to
    the `recurrent_kernel` weights matrix
    (see [constraints](https://keras.io/constraints/)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](https://keras.io/constraints/)).
- __dropout__: Float between 0 and 1.
    Fraction of the units to drop for
    the linear transformation of the inputs.
- __recurrent_dropout__: Float between 0 and 1.
    Fraction of the units to drop for
    the linear transformation of the recurrent state.
- __implementation__: Implementation mode, either 1 or 2.
    Mode 1 will structure its operations as a larger number of
    smaller dot products and additions, whereas mode 2 will
    batch them into fewer, larger operations. These modes will
    have different performance profiles on different hardware and
    for different applications.
- __return_sequences__: Boolean. Whether to return the last output
    in the output sequence, or the full sequence.
- __return_state__: Boolean. Whether to return the last state
    in addition to the output. The returned elements of the
    states list are the hidden state and the cell state, respectively.
- __go_backwards__: Boolean (default False).
    If True, process the input sequence backwards and return the
    reversed sequence.
- __stateful__: Boolean (default False). If True, the last state
    for each sample at index i in a batch will be used as initial
    state for the sample of index i in the following batch.
- __unroll__: Boolean (default False).
    If True, the network will be unrolled,
    else a symbolic loop will be used.
    Unrolling can speed-up a RNN,
    although it tends to be more memory-intensive.
    Unrolling is only suitable for short sequences.

Example:
```yaml

```
## GlobalMaxPool1D

Global max pooling operation for temporal data.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **data_format** - A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.  channels_last corresponds to inputs with shape  (batch, steps, features) while channels_first corresponds to inputs with shape  (batch, features, steps).

Example:
```yaml

```
## GlobalAveragePooling1D

Global average pooling operation for temporal data.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **data_format** - A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.  channels_last corresponds to inputs with shape  (batch, steps, features) while channels_first corresponds to inputs with shape  (batch, features, steps).

Example:
```yaml

```
## BatchNormalization

Batch normalization layer.

Normalize the activations of the previous layer at each batch,
i.e. applies a transformation that maintains the mean activation
close to 0 and the activation standard deviation close to 1.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
- __axis__: Integer, the axis that should be normalized
    (typically the features axis).
    For instance, after a `Conv2D` layer with
    `data_format="channels_first"`,
    set `axis=1` in `BatchNormalization`.
- __momentum__: Momentum for the moving mean and the moving variance.
- __epsilon__: Small float added to variance to avoid dividing by zero.
- __center__: If True, add offset of `beta` to normalized tensor.
    If False, `beta` is ignored.
- __scale__: If True, multiply by `gamma`.
    If False, `gamma` is not used.
    When the next layer is linear (also e.g. `nn.relu`),
    this can be disabled since the scaling
    will be done by the next layer.
- __beta_initializer__: Initializer for the beta weight.
- __gamma_initializer__: Initializer for the gamma weight.
- __moving_mean_initializer__: Initializer for the moving mean.
- __moving_variance_initializer__: Initializer for the moving variance.
- __beta_regularizer__: Optional regularizer for the beta weight.
- __gamma_regularizer__: Optional regularizer for the gamma weight.
- __beta_constraint__: Optional constraint for the beta weight.
- __gamma_constraint__: Optional constraint for the gamma weight.

Example:
```yaml

```
## Concatenate

Layer that concatenates a list of inputs.

Example:
```yaml
- concatenate: [lstmBranch,textFeatureBranch]
```
## Add

Layer that adds a list of inputs.

It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).

Example:
```yaml
- add: [first,second]
```
## Substract

ayer that subtracts two inputs.

It takes as input a list of tensors of size 2, both of the same shape, and returns a single tensor, (inputs[0] - inputs[1]), also of the same shape.

Example:
```yaml
- substract: [first,second]
```

## Mult

Layer that multiplies (element-wise) a list of inputs.

It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).

Example:
```yaml
- mult: [first,second]
```
## Max

Layer that computes the maximum (element-wise) a list of inputs.

It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).

Example:
```yaml
- max: [first,second]
```
## Min

Layer that computes the minimum (element-wise) a list of inputs.

It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).

Example:
```yaml
- min: [first,second]
```
## Conv1D

1D convolution layer (e.g. temporal convolution).

This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.

When using this layer as the first layer in a model, provide an input_shape argument (tuple of integers or None, does not include the batch axis), e.g. input_shape=(10, 128) for time series sequences of 10 time steps with 128 features per step in data_format="channels_last", or (None, 128) for variable-length sequences with 128 features per step.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
- __filters__: Integer, the dimensionality of the output space
    (i.e. the number of output filters in the convolution).
- __kernel_size__: An integer or tuple/list of a single integer,
    specifying the length of the 1D convolution window.
- __strides__: An integer or tuple/list of a single integer,
    specifying the stride length of the convolution.
    Specifying any stride value != 1 is incompatible with specifying
    any `dilation_rate` value != 1.
- __padding__: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
    `"valid"` means "no padding".
    `"same"` results in padding the input such that
    the output has the same length as the original input.
    `"causal"` results in causal (dilated) convolutions,
    e.g. `output[t]` does not depend on `input[t + 1:]`.
    A zero padding is used such that
    the output has the same length as the original input.
    Useful when modeling temporal data where the model
    should not violate the temporal order. See
    [WaveNet: A Generative Model for Raw Audio, section 2.1](
    https://arxiv.org/abs/1609.03499).
- __data_format__: A string,
    one of `"channels_last"` (default) or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, steps, channels)`
    (default format for temporal data in Keras)
    while `"channels_first"` corresponds to inputs
    with shape `(batch, channels, steps)`.
- __dilation_rate__: an integer or tuple/list of a single integer, specifying
    the dilation rate to use for dilated convolution.
    Currently, specifying any `dilation_rate` value != 1 is
    incompatible with specifying any `strides` value != 1.
- __activation__: Activation function to use
    (see [activations](https://keras.io/activations/)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix
    (see [initializers](https://keras.io/initializers/)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](https://keras.io/initializers/)).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](https://keras.io/regularizers/)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](https://keras.io/regularizers/)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](https://keras.io/regularizers/)).
- __kernel_constraint__: Constraint function applied to the kernel matrix
    (see [constraints](https://keras.io/constraints/)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](https://keras.io/constraints/)).

Example:
```yaml

```
## Conv2D

2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.

When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
- __filters__: Integer, the dimensionality of the output space
    (i.e. the number of output filters in the convolution).
- __kernel_size__: An integer or tuple/list of 2 integers, specifying the
    height and width of the 2D convolution window.
    Can be a single integer to specify the same value for
    all spatial dimensions.
- __strides__: An integer or tuple/list of 2 integers,
    specifying the strides of the convolution
    along the height and width.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Specifying any stride value != 1 is incompatible with specifying
    any `dilation_rate` value != 1.
- __padding__: one of `"valid"` or `"same"` (case-insensitive).
    Note that `"same"` is slightly inconsistent across backends with
    `strides` != 1, as described
    [here](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860)
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, height, width, channels)` while `"channels_first"`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Keras config file at `~/.keras/keras.json`.
    If you never set it, then it will be "channels_last".
- __dilation_rate__: an integer or tuple/list of 2 integers, specifying
    the dilation rate to use for dilated convolution.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Currently, specifying any `dilation_rate` value != 1 is
    incompatible with specifying any stride value != 1.
- __activation__: Activation function to use
    (see [activations](https://keras.io/activations/)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix
    (see [initializers](https://keras.io/initializers/)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](https://keras.io/initializers/)).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](https://keras.io/regularizers/)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](https://keras.io/regularizers/)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](https://keras.io/regularizers/)).
- __kernel_constraint__: Constraint function applied to the kernel matrix
    (see [constraints](https://keras.io/constraints/)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](https://keras.io/constraints/)).

Example:
```yaml

```
## MaxPool1D

Max pooling operation for temporal data.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
- __pool_size__: Integer, size of the max pooling windows.
- __strides__: Integer, or None. Factor by which to downscale.
    E.g. 2 will halve the input.
    If None, it will default to `pool_size`.
- __padding__: One of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, steps, features)` while `channels_first`
    corresponds to inputs with shape
    `(batch, features, steps)`.

Example:
```yaml

```
## MaxPool2D

Max pooling operation for spatial data.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
- __pool_size__: integer or tuple of 2 integers,
    factors by which to downscale (vertical, horizontal).
    (2, 2) will halve the input in both spatial dimension.
    If only one integer is specified, the same window length
    will be used for both dimensions.
- __strides__: Integer, tuple of 2 integers, or None.
    Strides values.
    If None, it will default to `pool_size`.
- __padding__: One of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, height, width, channels)` while `channels_first`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Keras config file at `~/.keras/keras.json`.
    If you never set it, then it will be "channels_last".

Example:
```yaml

```
## AveragePooling1D

Average pooling for temporal data.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
- __pool_size__: Integer, size of the average pooling windows.
- __strides__: Integer, or None. Factor by which to downscale.
    E.g. 2 will halve the input.
    If None, it will default to `pool_size`.
- __padding__: One of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, steps, features)` while `channels_first`
    corresponds to inputs with shape
    `(batch, features, steps)`.

Example:
```yaml

```
## CuDNNLSTM

Fast LSTM implementation with [CuDNN](https://developer.nvidia.com/cudnn).

Can only be run on GPU, with the TensorFlow backend.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
- __units__: Positive integer, dimensionality of the output space.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
    used for the linear transformation of the inputs.
    (see [initializers](https://keras.io/initializers/)).
- __recurrent_initializer__: Initializer for the `recurrent_kernel`
    weights matrix,
    used for the linear transformation of the recurrent state.
    (see [initializers](https://keras.io/initializers/)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](https://keras.io/initializers/)).
- __unit_forget_bias__: Boolean.
    If True, add 1 to the bias of the forget gate at initialization.
    Setting it to true will also force `bias_initializer="zeros"`.
    This is recommended in [Jozefowicz et al. (2015)](
    http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](https://keras.io/regularizers/)).
- __recurrent_regularizer__: Regularizer function applied to
    the `recurrent_kernel` weights matrix
    (see [regularizer](https://keras.io/regularizers/)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](https://keras.io/regularizers/)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](https://keras.io/regularizers/)).
- __kernel_constraint__: Constraint function applied to
    the `kernel` weights matrix
    (see [constraints](https://keras.io/constraints/)).
- __recurrent_constraint__: Constraint function applied to
    the `recurrent_kernel` weights matrix
    (see [constraints](https://keras.io/constraints/)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](https://keras.io/constraints/)).
- __return_sequences__: Boolean. Whether to return the last output.
    in the output sequence, or the full sequence.
- __return_state__: Boolean. Whether to return the last state
    in addition to the output.
- __stateful__: Boolean (default False). If True, the last state
    for each sample at index i in a batch will be used as initial
    state for the sample of index i in the following batch.

Example:
```yaml

```
## Dense

Regular densely-connected NN layer.

`Dense` implements the operation:
`output = activation(dot(input, kernel) + bias)`
where `activation` is the element-wise activation function
passed as the `activation` argument, `kernel` is a weights matrix
created by the layer, and `bias` is a bias vector created by the layer
(only applicable if `use_bias` is `True`).

Note: if the input to the layer has a rank greater than 2, then
it is flattened prior to the initial dot product with `kernel`.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
- __units__: Positive integer, dimensionality of the output space.
- __activation__: Activation function to use
    (see [activations](https://keras.io/activations/)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix
    (see [initializers](https://keras.io/initializers/)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](https://keras.io/initializers/)).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](https://keras.io/regularizers/)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](https://keras.io/regularizers/)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](https://keras.io/regularizers/)).
- __kernel_constraint__: Constraint function applied to
    the `kernel` weights matrix
    (see [constraints](https://keras.io/constraints/)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](https://keras.io/constraints/)).

Example:
```yaml

```
## Flatten

Flattens the input. Does not affect the batch size.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    The purpose of this argument is to preserve weight
    ordering when switching a model from one data format
    to another.
    `channels_last` corresponds to inputs with shape
    `(batch, ..., channels)` while `channels_first` corresponds to
    inputs with shape `(batch, channels, ...)`.
    It defaults to the `image_data_format` value found in your
    Keras config file at `~/.keras/keras.json`.
    If you never set it, then it will be "channels_last".

Example:
```yaml

```
## Bidirectional

Bidirectional wrapper for RNNs.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
- __layer__: `Recurrent` instance.
- __merge_mode__: Mode by which outputs of the
    forward and backward RNNs will be combined.
    One of {'sum', 'mul', 'concat', 'ave', None}.
    If None, the outputs will not be combined,
    they will be returned as a list.
- __weights__: Initial weights to load in the Bidirectional model

Example:
```yaml

```

# Utility layers

## split

Splits current flow into several ones.
Each child is a separate flow with an input equal to the input of the split operation.

Number of outputs is equal to a number of children.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## split-concat

Splits current flow into several ones.
Each child is a separate flow with an input equal to the input of the split operation.

Output is a concatenation of child flows.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml
- split-concat:
         - word_indexes_embedding:  [ embeddings/glove.840B.300d.txt ]
         - word_indexes_embedding:  [ embeddings/paragram_300_sl999.txt ]
         - word_indexes_embedding:  [ embeddings/wiki-news-300d-1M.vec]
- lstm2: [128]
```
## split-concatenate

Splits current flow into several ones.
Each child is a separate flow with an input equal to the input of the split operation.

Output is a concatenation of child flows (equal to the usage of [Concatenate](#concatenate) layer).

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml
- split-concat:
         - word_indexes_embedding:  [ embeddings/glove.840B.300d.txt ]
         - word_indexes_embedding:  [ embeddings/paragram_300_sl999.txt ]
         - word_indexes_embedding:  [ embeddings/wiki-news-300d-1M.vec]
- lstm2: [128]
```
## split-add

Splits current flow into several ones.
Each child is a separate flow with an input equal to the input of the split operation.

Output is an addition of child flows (equal to the usage of [Add](#add) layer).

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## split-substract

Splits current flow into several ones.
Each child is a separate flow with an input equal to the input of the split operation.

Output is a substraction of child flows (equal to the usage of [Substract](#substract) layer).

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## split-mult

Splits current flow into several ones.
Each child is a separate flow with an input equal to the input of the split operation.

Output is a multiplication of child flows (equal to the usage of [Mult](#mult) layer).

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## split-min

Splits current flow into several ones.
Each child is a separate flow with an input equal to the input of the split operation.

Output is a minimum of child flows (equal to the usage of [Min](#min) layer).


Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## split-max

Splits current flow into several ones.
Each child is a separate flow with an input equal to the input of the split operation.

Output is a maximum of child flows (equal to the usage of [Max](#max) layer).


Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## split-dot

Splits current flow into several ones.
Each child is a separate flow with an input equal to the input of the split operation.

Output is a dot product of child flows.


Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## split-dot-normalize

Splits current flow into several ones.
Each child is a separate flow with an input equal to the input of the split operation.

Output is a dot product with normalization of child flows.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## seq

Executes child elements as a sequence of operations, one by one.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## input

Overrides current input with what is listed.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml
input: [firstRef, secondRef]
```

## pass

Stops execution of this branch and drops its output.
TODO check that description is correct.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## transform-concat

TODO

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

```
## transform-add

TODO

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.
* **** - 

Example:
```yaml

``` 

# Stage properties

## loss

**type**: ``string`` 

Sets the loss name.

Uses loss name detection mechanism to search for the built-in loss or for a custom function with the same name across project modules.

Example:
```yaml
loss: binary_crossentropy
```
## lr

**type**: ``float`` 

Learning rate.

Example:
```yaml

```
## initial_weights
**type**: ``string`` 

Fil path to load stage NN initial weights from.

Example:
```yaml
initial_weights: /initial.weights
```
## epochs
**type**: ``integer`` 

Number of epochs to train for this stage.

Example:
```yaml

```
## unfreeze_encoder
TODO is this for generic?
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
## extra_callbacks
TODO

# Preprocessors
**type**: ``complex`` 

Preprocessors are the custom python functions that transform dataset. 

Such functions should be defined in python files that are in a project scope (`modules`) folder and imported.
Preprocessing functions should be also marked with `@preprocessing.dataset_preprocessor` annotation.

`Preprocessors` instruction then can be used to chain preprocessors as needed for this particular experiment, and even cache the result on disk to be reused between experiments.

Example:
```yaml
preprocessing: 
  - binarize_target: 
  - tokenize:  
  - tokens_to_indexes:
       maxLen: 160
  - disk-cache: 
```
## cache

Caches its input.
TODO what for?

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## disk-cache

Caches its input on disk, including the full flow. 
On subsequent launches if nothing was changed in the flow, takes its output from disk instead of re-launching previous operations. 

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml
preprocessing: 
  - binarize_target: 
  - tokenize:  
  - tokens_to_indexes:
       maxLen: 160
  - disk-cache: 
```
## split-preprocessor

An analogue of [split](#split) for preprocessor operations.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## split-concat-preprocessor

An analogue of [split-concat](#split-concat) for preprocessor operations.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
## seq-preprocessor

An analogue of [seq](#seq-concat) for preprocessor operations.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```

## augmentation

Preprocessor instruction, which body only runs during the training and is skipped when the inferring.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
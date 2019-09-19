# Segmentation pipeline reference

## Pipeline root properties

### activation
**type**: `string`

Activation function that should be used in last layer. In the case of binary segmentation it usually should be `sigmoid` if you have
more then one class than most likely you need to use `softmax`, but actually you are free to use any activation function that is
registered in Keras

Example:
```yaml
activation: sigmoid
```

### aggregation_metric

**type**: ``string`` 

Metric to calculate against the combination of all stages and report in `allStages` section of summary.yaml file after all experiment instances are finished.

Uses metric name detection mechanism to search for the built-in metric or for a custom function with the same name across project modules.

Metric name may have `val_` prefix or `_holdout` postfix to indicate calculation against validation or holdout, respectively. 

Example:
```yaml
aggregation_metric: matthews_correlation_holdout

```
### architecture

**type**: ``string`` 

This property configures decoder architecture that should be used:

At this moment segmentation pipeline supports following architectures:

- [Unet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [Linknet](https://codeac29.github.io/projects/linknet/)
- [PSP](https://arxiv.org/abs/1612.01105)
- [FPN](https://arxiv.org/abs/1612.03144)
- [DeeplabV3](https://arxiv.org/abs/1706.05587)

Example:
```yaml
architecture: FPN
```
### augmentation

**type**: ``complex`` 

[IMGAUG](https://imgaug.readthedocs.io) transformations sequence.
Each object is mapped on [IMGAUG](https://imgaug.readthedocs.io) transformer by name, parameters are mapped too.

Example:
```yaml
transforms:
 Fliplr: 0.5
 Affine:
   translate_px:
     x:
       - -50
       - +50
     y:
       - -50
       - +50
```
### backbone
**type**: ``string``

This property configures encoder that should be used:


`FPN`, `PSP`, `Linkenet`, `UNet` architectures support following backbones: 

  - [VGGNet](https://arxiv.org/abs/1409.1556)
    - vgg16
    - vgg19
  - [ResNet](https://arxiv.org/abs/1512.03385)
    - resnet18
    - resnet34
    - resnet50 
    - resnet101
    - resnet152
  - [ResNext](https://arxiv.org/abs/1611.05431)
    - resnext50
    - resnext101
  - [DenseNet](https://arxiv.org/abs/1608.06993)
    - densenet121
    - densenet169
    - densenet201
  - [Inception-v3](https://arxiv.org/abs/1512.00567)
  - [Inception-ResNet-v2](https://arxiv.org/abs/1602.07261)
  
All them support the weights pretrained on [ImageNet](http://www.image-net.org/):
```yaml
encoder_weights: imagenet
```

At this moment `DeeplabV3` architecture supports following backbones:
 - [MobileNetV2](https://arxiv.org/abs/1801.04381)
 - [Xception](https://arxiv.org/abs/1610.02357)

Deeplab supports weights pretrained on [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/):

### batch

**type**: ``integer`` 

Sets up training batch size.

Example:
```yaml
batch: 512
```

### classifier
**type**: `string` 


TODO description

Supported values:
- [ResNet](https://arxiv.org/abs/1512.03385)
    - resnet50
[DenseNet](https://arxiv.org/abs/1608.06993)
    - densenet121
    - densenet169
    - densenet201

Example:
```yaml

```
### classifier_lr
**type**: `float` 

TODO description

Example:
```yaml

```
### classes
**type**: `integer` 

Number of classes that should be segmented.

Example:
```yaml

```
### callbacks

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

### compressPredictionsAsInts
**type**: `boolean` 

Whether to represent predictions as integers (up to 4 channels)
TODO check this is correct

Example:
```yaml
compressPredictionsAsInts: true
```

### copyWeights

**type**: ``boolean`` 

Whether to copy saved weights.

Example:
```yaml
copyWeights: true
```
### clipnorm

**type**: ``float`` 

Maximum clip norm of a gradient for an optimizer.

Example:
```yaml
clipnorm: 1.0
```
### clipvalue

**type**: ``float`` 

Clip value of a gradient for an optimizer.

Example:
```yaml
clipvalue: 0.5
```

### crops
**type**: `integer` 

Defines the number of crops to make from original image by setting the single number of single dimension cells.
In example, the value of 3 will split the original image into 9 cells: 3 by horizontal and 3 by vertical.

Example:
```yaml
crops: 3
```

### dataset

**type**: ``complex object`` 

Key is a name of the python function in scope, which returns training data set.
Value is an array of parameters to pass to a function.

Example:
```yaml
dataset:
  getTrain: [false,false]
```
### datasets

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
### dataset_augmenter

**type**: ``complex object`` 

Sets up a custom augmenter function to be applied to a dataset.
Object must have a name property, whic will be used as a name of the python function in scope.
Other object properties are mapped as function arguments.

Example:
```yaml
dataset_augmenter:
    name: TheAugmenter
    parameter: test
```
### dropout

**type**: ``float`` 


Example:
```yaml

```
### encoder_weights
**type**: ``string`` 

This property configures initial weights of the encoder, supported values:

`imagenet`

Example:
```yaml
encoder_weights: imagenet
```

### extra_train_data

**type**: ``string`` 

Name of the additional dataset that will be added (per element) to the training dataset before train launching.

Example:
```yaml

```
### folds_count

**type**: ``integer`` 

Number of folds to train. Default is 5.

Example:
```yaml

```
### freeze_encoder

**type**: ``boolean`` 

Whether to freeze encoder during the training process.

Example:
```yaml
freeze_encoder: true
stages:
  - epochs: 10 #Let's go for 10 epochs with frozen encoder

  - epochs: 100 #Now let's go for 100 epochs with trainable encoder
    unfreeze_encoder: true  
```
### final_metrics

**type**: ``array of strings`` 

Metrics to calculate against every stage and report in `stages` section of summary.yaml file after all experiment instances are finished.

Uses metric name detection mechanism to search for the built-in metric or for a custom function with the same name across project modules.

Metric name may have `val_` prefix or `_holdout` postfix to indicate calculation against validation or holdout, respectively.

Example:
```yaml
final_metrics: [measure]

```
### holdout

**type**: ```` 


Example:
```yaml

```
### imports

**type**: ``array of strings`` 

Imports python files from `modules` folder of the project and make their properly annotated contents to be available to be referred from YAML.

Example:
```yaml
imports: [ layers, preprocessors ]

```
this will import `layers.py` and `preprocessors.py`
### inference_batch

**type**: ``integer`` 

Size of batch during inferring process.

Example:
```yaml

```
### loss

**type**: ``string`` 

Sets the loss name.

Uses loss name detection mechanism to search for the built-in loss or for a custom function with the same name across project modules.

Example:
```yaml
loss: binary_crossentropy
```
### lr

**type**: ``float`` 

Learning rate.

Example:
```yaml

```

### manualResize
**type**: `boolean` 

TODO

Example:
```yaml

```

### metrics

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
### num_seeds

**type**: ``integer`` 

If set, training process (for all folds) will be executed `num_seeds` times, each time resetting the random seeds.
Respective folders (like `metrics`) will obtain subfolders `0`, `1` etc... for each seed.

Example:
```yaml

```
### optimizer

**type**: ``string`` 

Sets the optimizer.

Example:
```yaml
optimizer: Adam
```
### primary_metric

**type**: `string` 

Metric to track during the training process. Metric calculation results will be printed in the console and to `metrics` folder of the experiment.

Besides tracking, this metric will be also used by default for metric-related activity, in example, for decision regarding which epoch results are better.

Uses metric name detection mechanism to search for the built-in metric or for a custom function with the same name across project modules.

Metric name may have `val_` prefix or `_holdout` postfix to indicate calculation against validation or holdout, respectively.

Example:
```yaml
primary_metric: val_macro_f1
```
### primary_metric_mode

**type**: ``enum: auto,min,max`` 

**default**: ``auto`` 

In case of a usage of a primary metrics calculation results across several instances (i.e. batches), this will be a mathematical operation to find a final result.

Example:
```yaml
primary_metric_mode: max
```
### preprocessing

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
### random_state

**type**: ``integer`` 

The seed of randomness.

Example:
```yaml

```

### shape

**type**: `array of integers` 

Shape of the input picture, in the form heigth,width, number of channels, all images will be resized to this shape before processing

Example:

```yaml
shape: [440,440,3]
```

### stages

**type**: ``complex`` 

Sets up training process stages. 
Contains YAML array of stages, where each stage is a complex type that may contain properties described in the [Stage properties](#stage-properties) section.  

Example:
```yaml
stages:
  - epochs: 6
  - epochs: 6
    lr: 0.01
    
```
### stratified

**type**: ``boolean`` 

Whether to use stratified strategy when splitting training set.

Example:
```yaml

```
### testSplit

**type**: `float 0-1` 

Splits the train set into two parts, using one part for train and leaving the other untouched for a later testing.
The split is shuffled.

Example:
```yaml
testSplit: 0.4
```
### testSplitSeed

**type**: ```` 

Seed of randomness for the split of the training set.

Example:
```yaml

```
### testTimeAugmentation

**type**: ``string`` 

Test-time augumentation function name.
Function must be reachable on project scope, accept and return numpy array.

Example:
```yaml

```
### transforms

**type**: ``complex`` 

If yes, why are we having pure IMGAUG in generic called just "transforms", maybe we should call it "imageTransforms" or simply "imgaug". 
Btw, isnt it crossing with preprocessing, maybe we should just create "imgaug" preprocessor with all these goodies inside? 

[IMGAUG](https://imgaug.readthedocs.io) transformations sequence.
Each object is mapped on [IMGAUG](https://imgaug.readthedocs.io) transformer by name, parameters are mapped too.

Example:
```yaml
transforms:
 Fliplr: 0.5
 Affine:
   translate_px:
     x:
       - -50
       - +50
     y:
       - -50
       - +50
```
### validationSplit

**type**: ``float`` 

Float 0-1 setting up how much of the training set (after holdout is already cut off) to allocate for validation.

Example:
```yaml

```

## Callback types

### EarlyStopping

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

### ReduceLROnPlateau

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

### CyclicLR

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

### LRVariator

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


Example
```yaml
```

### TensorBoard

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

## Stage properties

### callbacks
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

### epochs
**type**: ``integer`` 

Number of epochs to train for this stage.

Example:
```yaml

```
### extra_callbacks

### freeze_encoder
**type**: ``boolean`` 

Whether to freeze encoder during the training process.

Example:
```yaml
freeze_encoder: true
stages:
  - epochs: 10 #Let's go for 10 epochs with frozen encoder

  - epochs: 100 #Now let's go for 100 epochs with trainable encoder
    unfreeze_encoder: true  
```

### initial_weights
**type**: ``string`` 

Fil path to load stage NN initial weights from.

Example:
```yaml
initial_weights: /initial.weights
```
### negatives

**type**: `string or integer` 

The support of binary data balancing for training set.

Following values are acceptable:

- none - exclude negative examples from the data
- real - include all negative examples 
- integer number(1 or 2 or anything), how many negative examples should be included per one positive example

In order for the system to determine whether a particular example is positive or negative,
the data set class defined by the [dataset](#dataset) property should have `isPositive` method declared 
that accepts data set item and returns boolean.

Example:
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
### loss

**type**: ``string`` 

Sets the loss name.

Uses loss name detection mechanism to search for the built-in loss or for a custom function with the same name across project modules.

Example:
```yaml
loss: binary_crossentropy
```
### lr

**type**: ``float`` 

Learning rate.

Example:
```yaml

```

### unfreeze_encoder
**type**: ``boolean`` 

Whether to unfreeze encoder during the training process.

Example:
```yaml
freeze_encoder: true
stages:
  - epochs: 10 #Let's go for 10 epochs with frozen encoder

  - epochs: 100 #Now let's go for 100 epochs with trainable encoder
    unfreeze_encoder: true  
```

### validation_negatives

**type**: `string or integer` 

The support of binary data balancing for validation set.

Following values are acceptable:

- none - exclude negative examples from the data
- real - include all negative examples 
- integer number(1 or 2 or anything), how many negative examples should be included per one positive example

In order for the system to determine whether a particular example is positive or negative,
the data set class defined by the [dataset](#dataset) property should have `isPositive` method declared 
that accepts data set item and returns boolean.

Example:
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


## Preprocessors
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
### cache

Caches its input.

Properties:

* **name** - string; optionally sets up layer name to refer it from other layers.
* **inputs** - array of strings; lists layer inputs.

Example:
```yaml

```
### disk-cache

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
### split-preprocessor

An analogue of [split](#split) for preprocessor operations.


Example:
```yaml

```
### split-concat-preprocessor

An analogue of [split-concat](#split-concat) for preprocessor operations.


Example:
```yaml

```
### seq-preprocessor

An analogue of [seq](#seq-concat) for preprocessor operations.


Example:
```yaml

```

### augmentation

Preprocessor instruction, which body only runs during the training and is skipped when the inferring.


```yaml
augmentation:
 Fliplr: 0.5
 Affine:
   translate_px:
     x:
       - -50
       - +50
     y:
       - -50
       - +50
```

In this example, `Fliplr` key is automatically mapped on [Fliplr agugmenter](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_flip.html),
their `0.5` parameter is mapped on the first `p` parameter of the augmenter.
Named parameters are also mapped, in example `translate_px` key of `Affine` is mapped on `translate_px` parameter of [Affine augmenter](https://imgaug.readthedocs.io/en/latest/source/augmenters.html?highlight=affine#affine).

## fit script arguments

### fit.py project

**type**: ``string`` 

Folder to search for experiments, project root.

Example:

`fit.py --project "path/to/project"`

### fit.py name

**type**: ``string or comma-separated list of strings`` 

Name of the experiment to launch, or a list of names.

Example:

`fit.py --name "experiment_name"`

`fit.py --name "experiment_name1, experiment_name2"`

### fit.py num_gpus

**type**: ``integer``

Default: 1

Number of GPUs to use during experiment launch.

Example:
`fit.py --num_gpus=1`

### fit.py gpus_per_net

**type**: ``integer`` 

Default: 1

Maximum number of GPUs to use per single experiment.

Example:
`fit.py --gpus_per_net=1`

### fit.py num_workers

**type**: ``integer`` 

Default: 1

Number of workers to use.

Example:
`fit.py --num_workers=1`

### fit.py allow_resume

**type**: ``boolean`` 

Default: False

Whether to allow resuming of experiments, 
which will cause unfinished experiments to start from the best saved weights.

Example:
`fit.py --allow_resume True`

### fit.py force_recalc

**type**: ``boolean`` 

Default: False

Whether to force rebuilding of reports and predictions.

Example:
`fit.py --force_recalc True`

### fit.py launch_tasks

**type**: ``boolean`` 

Default: False

Whether to launch associated tasks.

Example:
`fit.py --launch_tasks True`

### fit.py only_report

**type**: ``boolean`` 

Default: False

Whether to only generate reports for cached data, no training occurs.

Example:
`fit.py --only_report True`

### fit.py cache

**type**: ``string`` 

Path to the cache folder. 
Cache folder will contain temporary cached data for executed experiments.

Example:
`fit.py --cache "path/to/cache/folder"`

### fit.py folds

**type**: ``integer or comma-separated list of integers`` 

Folds to launch. By default all folds of experiment will be executed, 
this argument allows launching only some of them. 

Example:
`fit.py --folds 1,2`

### fit.py time

**type**: ``string`` 

TODO 

Example:
`fit.py `

## task script arguments

### task.py project

**type**: ``string`` 

Folder to search for experiments, project root.

Example:

`task.py --project "path/to/project"`

### task.py name

**type**: ``string or comma-separated list of strings`` 

Name of the experiment to launch, or a list of names.

Example:

`task.py --name "experiment_name"`

`task.py --name "experiment_name1, experiment_name2"`

### task.py task

**type**: ``string or comma-separated list of strings`` 

Default: all tasks.

Name of the task to launch, or a list of names.

Example:

`task.py --task "task_name"`

`task.py --task "task_name1, task_name2"`

`task.py --task "all"`

### task.py num_gpus

**type**: ``integer``

Default: 1

Number of GPUs to use during experiment launch.

Example:
`task.py --num_gpus=1`

### task.py gpus_per_net

**type**: ``integer`` 

Default: 1

Maximum number of GPUs to use per single experiment.

Example:
`task.py --gpus_per_net=1`

### task.py num_workers

**type**: ``integer`` 

Default: 1

Number of workers to use.

Example:
`task.py --num_workers=1`

### task.py allow_resume

**type**: ``boolean`` 

Default: False

Whether to allow resuming of experiments, 
which will cause unfinished experiments to start from the best saved weights.

Example:
`task.py --allow_resume True`

### task.py force_recalc

**type**: ``boolean`` 

Default: False

Whether to force rebuilding of reports and predictions.

Example:
`task.py --force_recalc True`

### task.py launch_tasks

**type**: ``boolean`` 

Default: False

Whether to launch associated tasks.

Example:
`task.py --launch_tasks True`

### task.py cache

**type**: ``string`` 

Path to the cache folder. 
Cache folder will contain temporary cached data for executed experiments.

Example:
`task.py --cache "path/to/cache/folder"`

## analyze script arguments

### analyze.py inputFolder

**type**: ``string`` 

Folder to search for finished experiments in. Typically, project root.

Example:

`analyze.py --inputFolder "path/to/project"`

### analyze.py output

**type**: ``string`` 

Default: `report.csv` in project root.

Output report file path.

Example:

`analyze.py --output "path/to/project/report/report.scv"`

### analyze.py onlyMetric

**type**: ``string`` 

Name of the single metric to take into account.

Example:

`analyze.py --onlyMetric "metric_name"`

### analyze.py sortBy

**type**: ``string`` 

Name of the metric to sort result by.

Example:

`analyze.py --sortBy "metric_name"`

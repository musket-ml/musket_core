# Instance Segmentation pipeline reference

## Pipeline root properties

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

### classes
**type**: `integer` 

Number of classes that should be segmented.

Example:
```yaml

```

### configPath

**type**: ``string``

Path to MMDetection config file. Should be absolute or relative to the musket config file.

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

### folds_count

**type**: ``integer`` 

Number of folds to train. Default is 5.

Example:
```yaml

```

### holdout

**type**: ```` 


Example:
```yaml

```

### imagesPerGpu
**type**: `integer` 

Number of images in a batch to be processed by single GPU.

MMDetection does not allow specifying batch size directly,
it only allows setting how much images are processed by each GPU at a time.
Thus, batch size is `imagesPerGpu` multiplied by [`gpus_per_net`](#fitpy-gpus_per_net).

Example:
```yaml
imagesPerGpu: 2
```

### imports

**type**: ``array of strings`` 

Imports python files from `modules` folder of the project and make their properly annotated contents to be available to be referred from YAML.

Example:
```yaml
imports: [ layers, preprocessors ]

```
this will import `layers.py` and `preprocessors.py`

### multiscaleMode

**type**: ``string``

Can be `range` or `value` (default). Setting value to `range` allows using two dimensional integer arrays as [shape](#shape)
values for specifying possible ranges of train shapes.

### num_seeds

**type**: ``integer`` 

If set, training process (for all folds) will be executed `num_seeds` times, each time resetting the random seeds.
Respective folders (like `metrics`) will obtain subfolders `0`, `1` etc... for each seed.

Example:
```yaml

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

**type**: one or two dimensional `array of integers`

Shape of the model input. All images are automatically scaled to this shape before being processed by the model. The exact meaning of the parameter can be:

* One dimensional array is simply understood as as `[height, width]` for train, validation and infering shapes.

* Two dimensional array is understood as array of shapes. Train shape is chosen randomly from the array for each train sample, and the first shape is always taken on validation and infering.

* With the [multiscaleMode](#multiscalemode) parameter set to `range` a two element two dimensional array `[[h1,w1], [h2,w2]]` is understood as possible range for train shapes:
train height and width are randomly chosen from `[h1, h2]` and `[w1, w2]` intervals respectively. Like in the previous case, the first shape is always taken on validation and infering.  


```yaml
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

### validationSplit

**type**: ``float`` 

Float 0-1 setting up how much of the training set (after holdout is already cut off) to allocate for validation.

Example:
```yaml

```

### weightsPath

**type**: ``string``

Path to the model pretrained weights. Should be absolute or relative to the musket config file.

### resetHeads

Whether to refuse adopting pretrained weights for mask head and bounding box head.   

Defaults to 'True'.

## Stage properties

### epochs
**type**: ``integer`` 

Number of epochs to train for this stage.

Example:
```yaml

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

`-m musket_core.fit --project "path/to/project"`

### fit.py name

**type**: ``string or comma-separated list of strings`` 

Name of the experiment to launch, or a list of names.

Example:

`-m musket_core.fit --name "experiment_name"`

`-m musket_core.fit --name "experiment_name1, experiment_name2"`

### fit.py num_gpus

**type**: ``integer``

Default: 1

Number of GPUs to use during experiment launch.

Example:
`-m musket_core.fit --num_gpus=1`

### fit.py gpus_per_net

**type**: ``integer`` 

Default: 1

Maximum number of GPUs to use per single experiment.

Example:
`-m musket_core.fit --gpus_per_net=1`

### fit.py num_workers

**type**: ``integer`` 

Default: 1

Number of workers to use.

Example:
`-m musket_core.fit --num_workers=1`

### fit.py allow_resume

**type**: ``boolean`` 

Default: False

Whether to allow resuming of experiments, 
which will cause unfinished experiments to start from the best saved weights.

Example:
`-m musket_core.fit --allow_resume True`

### fit.py force_recalc

**type**: ``boolean`` 

Default: False

Whether to force rebuilding of reports and predictions.

Example:
`-m musket_core.fit --force_recalc True`

### fit.py launch_tasks

**type**: ``boolean`` 

Default: False

Whether to launch associated tasks.

Example:
`-m musket_core.fit --launch_tasks True`

### fit.py only_report

**type**: ``boolean`` 

Default: False

Whether to only generate reports for cached data, no training occurs.

Example:
`-m musket_core.fit --only_report True`

### fit.py cache

**type**: ``string`` 

Path to the cache folder. 
Cache folder will contain temporary cached data for executed experiments.

Example:
`-m musket_core.fit --cache "path/to/cache/folder"`

### fit.py folds

**type**: ``integer or comma-separated list of integers`` 

Folds to launch. By default all folds of experiment will be executed, 
this argument allows launching only some of them. 

Example:
`-m musket_core.fit --folds 1,2`



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

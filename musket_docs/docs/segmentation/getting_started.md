### Requirements
TODO
### Getting dataset from Kaggle
#### Installing kaggle stuff 
This should be done only once.

Run `pip install kaggle` in console.

Log into [Kaggle](https://www.kaggle.com)
Click on a profile in the top-right corner and choose `My Account`

On the account page find `Api` section and click `Create New API Token`. 
This will launch the download of `kaggle.json` token file.
Put the file into `~/.kaggle/kaggle.json` or `C:\Users\<Windows-username>\.kaggle\kaggle.json` depending on OS.

Note: there are potential troubles of creating `C:\Users\<Windows-username>\.kaggle` using windows explorer. 
To create this folder from console, run `cmd` and launch the following commands:
`cd C:\Users\<Windows-username>`, `mkdir .kaggle`.

Consult to [Kaggle API](https://github.com/Kaggle/kaggle-api) in case of other troubles.

#### Downloading TGS Salt competition dataset

Go to [TGS Salt Identification competition](https://www.kaggle.com/c/tgs-salt-identification-challenge/rules) and Accept the rules on the `Rules` tab.

Make `salt` folder somewhere and create `data` subdirectory. Open console with `salt/data` folder as current 
and invoke `kaggle competitions download -c tgs-salt-identification-challenge` command.

This will download dataset files.
Then invoke `unzip train.zip -d train` to unzip `train.zip` files in to `train` folder.

### Adding an experiment

Create `experiments` folder inside `salt` folder.

Create `exp01` folder inside `experiments` folder.

Create `config.yaml` file inside `exp01` folder.

Put the following code inside `config.yaml`:

```yaml
#%Musket Segmentation 1.0
backbone: resnet34 #let's select classifier backbone for our network 
architecture: Unet #pre-trained model we are going to use
augmentation: #define some minimal augmentations on images
   Fliplr: 0.5
   Flipud: 0.5

classes: 1 #define the number of classes
activation: sigmoid #as we have multilabel classification, the activation for last layer is sigmoid
shape: [224,224, 3] #our desired input image size, everything will be resized to fit
optimizer: Adam #Adam optimizer is a good default choice
batch: 8 #our batch size will be 16
lr: 0.001 
metrics: #we would like to track some metrics
  - binary_accuracy
  - dice
primary_metric: val_dice #the most interesting metric is val_binary_accuracy
primary_metric_mode: max
folds_count: 5
testSplit: 0.2
dumpPredictionsToCSV: true
callbacks: #configure some minimal callbacks
  EarlyStopping:
    patience: 10
    monitor: val_dice
    mode: max
    verbose: 1
  ReduceLROnPlateau:
    patience: 2
    factor: 0.3
    monitor: val_binary_accuracy
    mode: max
    cooldown: 1
    verbose: 1
loss: binary_crossentropy #we use binary_crossentropy loss
stages:
  - epochs: 50
    
    
dataset:
   getTrain: []
final_metrics: [ dice_with_custom_treshold_true_negative_is_one ]   #You may use more then one metric here
experiment_result: dice_with_custom_treshold_true_negative_is_one     
testTimeAugmentation: Horizontal_and_vertical
```

You can find the details regarding this code in [User guide](index.md#general-train-properties).

We can greatly speed up the training process by reducing the 
number of folds from 5 to 1 by replacing `folds_count: 5` with 
`folds_count: 1`, but this will train the only fold.

Reducing the number of epochs will also speed things up by the cost of 
the quality: replace `  - epochs: 50` with `  - epochs: 20` if you wish so.

### Adding dataset

Note the following instruction in our experiment YAML:

```yaml
dataset:
   getTrain: []
``` 

This instruction expects a python function somewhere on the scope, which is named
`getTrain` and that should return dataset. Lets add it:

Create `modules` folder inside `salt` folder.

In `modules` folder create a file `datasets.py` (file name can be really anything).

Put the following code in the file:

```python
from musket_core import image_datasets

def getTrain():
    return image_datasets.BinarySegmentationDataSet(["train/images"],"train.csv","id","rle_mask")
```

First argument sets the images folder inside `data`.

Second argument points to the CSV, third - CSV column with image IDs.

The forth one points to CSV column with RLE mask.

### Running the experiment

Launch the console and run the following command, taking into account
that `..salt` should be replaced with the path to the project top-level
`salt` directory.
`musket fit --project "...salt" --name "exp01" --num_gpus=1 --gpus_per_net=1 --num_workers=1 --cache "...salt\data\cache"`

This will launch the training process.

### Checking experiment results

When the training process complete, `exp01` experiment folder will contain the 
new `summary.yaml` file with contents similar to the following:

```yaml
allStages:
  binary_accuracy: {max: 0.94829979903931, mean: 0.9432402460543085, min: 0.9348129166258212,
    std: 0.004766299735936636}
  binary_accuracy_holdout: 0.9447703656504265
  dice: {max: 0.8946749116884245, mean: 0.8799628551168702, min: 0.8512279018375117,
    std: 0.015545121737934034}
  dice_holdout: 0.8792530163407674
  dice_with_custom_treshold_true_negative_is_one: {max: 0.7921144300184988, mean: 0.7906020228394381,
    min: 0.7885881826799147, std: 0.0012594068050284218}
  dice_with_custom_treshold_true_negative_is_one_holdout: 0.7918140261625017
  dice_with_custom_treshold_true_negative_is_one_treshold: {max: 0.5700000000000001,
    mean: 0.5680000000000001, min: 0.56, std: 0.0040000000000000036}
  dice_with_custom_treshold_true_negative_is_one_treshold_holdout: 0.56
cfgName: config.yaml
completed: true
folds: [0, 1, 2, 3, 4]
stages:
- binary_accuracy: {max: 0.94829979903931, mean: 0.9432402460543085, min: 0.9348129166258212,
    std: 0.004766299735936636}
  binary_accuracy_holdout: 0.9447703656504265
  dice: {max: 0.8946749116884245, mean: 0.8799628551168702, min: 0.8512279018375117,
    std: 0.015545121737934034}
  dice_holdout: 0.8792530163407674
  dice_with_custom_treshold_true_negative_is_one: {max: 0.7924899348384953, mean: 0.7882949615678969,
    min: 0.7825054457176929, std: 0.003500868603109435}
  dice_with_custom_treshold_true_negative_is_one_holdout: 0.7918140261625017
  dice_with_custom_treshold_true_negative_is_one_treshold: {max: 0.6, mean: 0.5680000000000001,
    min: 0.51, std: 0.03310589071449368}
  dice_with_custom_treshold_true_negative_is_one_treshold_holdout: 0.56
subsample: 1.0

```

Lets take a closer look:

`completed: true` indicates that the training was completed.
Sections of `allStages` and `stages` differ when 
there are more than a single stage, but in our case data inside are the same.

`binary_accuracy` and `dice` values indicate metric results on validation.
Those values appear in summary due to those metrics were listed in `metrics`
section of `config.yaml`.

As we have multiple folds and each fold has its own results, all metrics list
max, min, mean and std values.

Due to the `testSplit` instruction in `config.yaml` we have a holdout,
so there is `*_holdout` values for each metrics indicating metric results
on holdout dataset.

As `dice` metric was referred in `primary_metric` instruction in `config.yaml`,
there are lots of other values for this metric, their names speak for themselves.

Besides `summary.yaml` file, which contain the top-level results and
may ommitted if something fails in the training process, there are more
detailed and precises logs inside `metrics` subfolder of `exp01` folder.

There are files named `metrics-X.Y.csv`, where `X` is the fold number, 
and `Y` is the stage number.

Lets take a look:
```csv
epoch,binary_accuracy,dice,loss,lr,val_binary_accuracy,val_dice,val_loss
0,0.8144591341726481,0.5552861321416458,0.4154038934037089,0.001,0.7323275666683913,0.026507374974244158,0.7329991146922111
1,0.8632630387321114,0.6899019529577345,0.331550125265494,0.001,0.7443207234144211,0.012612525901568005,0.5845782220363617
``` 

The first column is an epoch number.
Then there are all metrics listed on training set, then loss and learning rate, and finally same metrics and loss on validation.

These values allow to see how the training was advancing.

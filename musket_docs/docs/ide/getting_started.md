# Getting started
### Download
[Download Musket IDE for Windows](http://onpositive.club/public/dsideNew2.zip) 

[Download Musket IDE for MacOS](http://onpositive.club/public/dside.zip)
### Installation

* Install [Musket Core](../generic/index.md). Read installation instructions [here](../generic/index.md#installation).
* Install [Segmentation Pipeline](../segmentation/index.md). Read installation instructions [here](../segmentation/index.md#installation).
* Install [Classification Pipeline](../classification/index.md). Read installation instructions [here](../classification/index.md#installation).

Unzip and launch executable (`ds-ide.exe` for Windows and `ds-ide` for MacOS).

### Watch this in action

Most of the following contents can be also checked in action in [this video](https://www.youtube.com/watch?v=Hck1CH49-YQ&list=PLyV40LHl22j5VOv2DgHoNDkpxkfueIQmn&index=5&t=0s).

### Setting up a project

Launch `File->New...` from the main menu.

Choose `Musket->Musket Project` in the `New` dialog.

![New Project](images/NewProject.png)

Click `Next`, enter new project name and click `Please configure an interpreter...` to set up python interpreter.

Try `Config first in PATH` option first, if it fail o auto-detect you python, use other options.

![Project details](images/ProjectDialog.png)

Click `Finish` and accept opening the new perspective.

This will create the project structure for you, details regarding the structure can be found [here](../generic/index.md#project-structure).

### Getting dataset from Kaggle

#### Installing kaggle stuff 
This should be done only once, first time you are getting something from Kaggle.

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

#### Downloading the dataset

Back to IDE. Select your new project root in project explorer, right-click and select `New->Other`...
![new data set](images/NewDataSet1.png)

Choose `Musket->Kaggle Dataset`.

![new data set 2](images/NewDataSet2.png)

Click `Next` and select `competition`, enter `salt` into a search box and click `Search`.

![salt](images/SaltCompetition.png)

Select `tgs-salt-identification-chellenge` and click `Finish`.

This will start dataset download, check its progress in console.

### Analysing dataset

In project explorer, double-click on the newly downloaded `train.csv`

![train](images/TrainCSV.png)

Dataset editor displays CSV contents statistics, if the data is recognizable.

![DatasetStatistics](images/DatasetStatistics.png)

`Analyzers` combo allows to switch between dataset analysis modes.

![Analyzers](images/Analyzers.png)

In this simple binary segmentation there is an only suitable analyzer, 
but in other uses cases there are more.

So, the statistics on the screenshot is the output of the analyzer. 
Besides that, analyzer also splits and sorts data, and we can see that on `data` tab.

Visualizers are used to display that data, in our case, as we got raw images, there is an only visualizer.

![DatasetData](images/DatasetData.png)

Masks are also displayed.

### Creating an experiment

Following should be done to make an experiment from the dataset:

Click the `Generate` button in the toolbar in the top-right corner of the viewer.

![generate](images/Generate.png)

Choose a name like `train_ds` and `Generate Musket Wrappers`, click `Ok`.

![GenerateWrappers](images/GenerateWrappers.png)

Accept to configure an experiment, enter any name like `exp01` and click `Finish`.

You now have `exp01` folder inside `experiments` folder and `config.yaml` file there.

`datasets.py` is also generated inside `modules` folder.

### Editing an experiment

We've got a default experiment for binary segmentation generated in the previous chapter.

You can always find it in project explorer:

![ProjectExplorerExperiment](images/ProjectExplorerExperiment.png)

Its contents can be edited by double-clicking on the experiment.
![experiment](images/ExpFile.png)

Lets make some minor changes to experiment code.

Change the shape to be well eaten by the network to:

Change `shape` instruction for `shape: [224,224, 3]`

Lets reduce the count of folds from default 5 to 3 to speed things up:

Add `folds_count: 3` instruction to the root level.

Add some holdout so we can track the effectiveness of the trained experiment:

Add `testSplit: 0.2` instruction to the root level.

And add an instructions to dump validation and holdout predictions to CSV right after the training.

Add `dumpPredictionsToCSV: true` instruction to the root level.

### Running an experiment

Click on `Launch experiment` in the tool bar of the editor.

![Launch experiment](images/launch_experiment.png)

This should launch an experiment, you can track what's going on in the console.

### Checking results and logs

When the experiment is finished, overal statistics should appear in the `Results and logs` tab:

![Experiment results](images/ExperimentResults.png)

It must also generate `summary.yaml` file in the experiment folder and `metrics*.csv` files in `metrics` subfolder.

Check these files, they contain lots of useful statistics.

`Logs` tab display logs in graphical form.

![BinaryLogs](images/BinaryLogs.png)

`Log` combo switches between different logs we got, in our case there is one per fold.


![LogCombo](images/LogCombo.png)

We've got following metrics declared in the experiment YAML code:

```yaml
metrics: #we would like to track some metrics
  - binary_accuracy
  - dice
```

So the `Metric` combo lists all of them (for the specific fold) and loss:

![MetricsCombo](images/MetricsCombo.png)

### Checking predictions

As we dumped our validation and holdout predictions to CSV by adding 
`dumpPredictionsToCSV: true` instruction to the root level of our YAML, 
now we have a bunch of CSV files in the `predictions` folder of our experiment.

You can use those files directly, our just click the links that appear on the `Results and logs`/`Results` tab:

![PredictionLinks](images/PredictionLinks.png)

This opens up the viewer with visualizers and analyzers that we already seen,
this time for particular prediction.

Statistics tab, as usual, displays some chart.
![PredictionStatistics](images/PredictionStatistics.png)

Check out `Analyzers` combo, this time there are more of them.
![PredictionStatistics2](images/PredictionStatistics2.png)

`Data` tab, which grouping is affected by the current analyzer displays samples with mask and prediction.
![PredictionData](images/PredictionData.png)

### What is next

Check the videos to find out what else can be done using Musket IDE: [tutorials](https://www.youtube.com/playlist?list=PLyV40LHl22j5VOv2DgHoNDkpxkfueIQmn).
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

Make `salt` folder somewhere and create `dataset` subdirectory. Open console with `salt/dataset` folder as current 
and invoke `kaggle competitions download -c tgs-salt-identification-challenge` command.

This will download dataset files.
Then invoke `unzip train.zip -d train` to unzip `train.zip` files in to `train` folder.


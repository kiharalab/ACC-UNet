# Experiments

Our experimental codes are based on 

https://github.com/McGregorWwww/UCTransNet

We thankfully acknowledge the contributions of the authors

<hr>

In order to run the experiments, please follow these steps

## Prepare the Dataset

Prepare the train, validation and test split of your dataset.   

Then store all the images and corresponding masks as .png images and save with the same name, in the img and labelcol directories, respectively.

The directory structure should be as follows:

```angular2html
├── datasets
    ├── GlaS_exp1
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    └── BUSI_exp1
        ├── Test_Folder
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── img
            └── labelcol
```

## Train the Model

First, modify the model, dataset and training hyperparameters in `Config.py`

Then simply run the training code.

```
python3 train_model.py
```

**Note:** In order to train Swin-Unet or SMESwin-UNet please collect the pretrained checkpoint from https://github.com/HuCaoFighting/Swin-Unet and put that inside `Experiments/pretrained_ckpt`


## Evaluate the Model

Please make sure the right model and dataset is selected in `Config.py`

Then simply run the evaluation code.

```
python3 test_model.py
```

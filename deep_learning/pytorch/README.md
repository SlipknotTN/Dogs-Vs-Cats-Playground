# PyTorch Dogs Vs Cats example

Example usage of PyTorch to classify dogs and cats images
(from Kaggle challenge [Dogs Vs Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)).

Tested with python3 and PyTorch 0.4.1.

## Models

### SqueezeNet 1.1

Model pretrained on ImageNet provided by PyTorch, trained from scratch only the last convolution layer to classify the target classes.
[[Paper]](https://arxiv.org/abs/1704.04861).

**Image format:** RGB.

**Image preprocessing during training:** shorter side resized to 256, extracted random crop of 224 x 224,
PyTorch pretrained models preprocessing: images loaded in to a range of [0, 1] and then normalized
using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
You can explore other preprocessing steps

**Image preprocessing during validation and test:** resized to 224 x 224,
PyTorch pretrained models preprocessing: images loaded in to a range of [0, 1] and then normalized
using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

### Kaggle Results

| Model | Input Size | HyperParameters | Validation accuracy | Kaggle LogLoss|
|-------|------------|-----------------|---------------|------------------|
| SqueezeNet 1.1 | 224x224 | 20 epochs, LR 0.001, Momentum 0.9, SGD optimizer | 0.9602 | 0.15168 |

### Custom model

TODO: Create a custom architecture showing PyTorch API

## Training instructions

Required steps:
- Split dataset training directory in train and validation subsets
- Fine tune training of the pretrained model on dogs vs cats classification task 

The directory *config* contains training parameters for each model.
You only need to pass a different config to the scripts to change model and hyperparameters.

### Split dataset subsets

The script considers that *../../kaggle_dataset* (kaggle_dataset in repo root) contains the extracted dogs vs cats kaggle dataset.
In particular *train* directory must contain all labelled dataset images.

Run this script from *repo root* directory.

```
python3 ./dataset/split.py
--trainDatasetDir ./kaggle_dataset/train
--valSplit 20
--outputDatasetDir ./kaggle_dataset/trainval
```

After the script execution with these arguments you will get this subdirectories in *kaggle_dataset* directory:
- trainval/train: symlinks to train subset images
- trainval/val: symlinks to validation subset images (80/20 split)

### Training

With squeezenet pretrained model we create a new branch in the architecture for dogs vs cats classification task
(adding only a new layer trained from scratch) and keeping old layers frozen.
With custom model we train all the layers from scratch.

#### Training examples

Run script from *pytorch* directory.

SqueezeNet 1.1:

```
python3 ./tools/train.py
--dataset_train_dir ../,,/kaggle_dataset/trainval/train
--dataset_val_dir ../../kaggle_dataset/trainval/val
--config_file ./config/sqnet.cfg
--model_output_path ./export/sqnet.pth
```

Custom model:

TODO

Export directory will contains this models:
- PyTorch format (single pth file with model weights).


### [Optional] Run predictions on test directory

This script runs predictions on dogs vs cats test images and exports results in a CSV format ready for kaggle submission.
Please notice that while the competition is over, you can still evaluate your model through [challenge page](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/leaderboard).

#### Test examples

Run script from *pytorch* directory.

SqueezeNet 1.1:

```
python3 ./tools/test_dir_for_export.py
--dataset_test_dir ../../kaggle_dataset/test
--config_file ./config/config_sqnet.cfg
--model_path ./export/sqnet.pth
--kaggle_export_file ./export/sqnet.csv
```

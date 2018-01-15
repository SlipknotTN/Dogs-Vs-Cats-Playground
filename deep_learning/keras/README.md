# Keras Dogs Vs Cats example

Example usage of keras to classify dogs and cats images (from Kaggle challenge [Dogs Vs Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)).

Tested with python3 and keras2 with tensorflow backend 1.3.

## Models

### MobileNet

Mobilenet pretrained on ImageNet, trained from scratch only the last convolution layer to classify the target classes. [[Paper]](https://arxiv.org/abs/1704.04861) [[Official TF slim models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

**Image format:** RGB.

**Image preprocessing:** transform from [0, 255] uint8 range to [-1.0, 1.0] float (Inception preprocessing).

### Kaggle Results

| Model | Input Size | HyperParameters | Validation accuracy | Kaggle LogLoss|
|-------|------------|-----------------|---------------|------------------|
| MobileNet 1.0  | 224x224 | 20 epochs, LR 0.001, Momentum 0.9, SGD optimizer | 0.9795 | 0.08296 |
| MobileNet 0.25 | 128x128 | 20 epochs, LR 0.001, Momentum 0.9, SGD optimizer | 0.9049 | 0.24903 |

### Custom model

Example usage of Sequential API to define a new architectur and train it from scratch.
TODO: Improve the model and test on kaggle.

## Training instructions

Required steps:
- Split dataset training directory in train and validation subsets
- Fine tune training of the pretrained model on dogs vs cats classification task 

The directory *config* contains training parameters for each model. You only need to pass a different config to the scripts to change model and hyperparameters.

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

### Fine tuning training

With mobilenet pretrained model we create a new branch in the architecture for dogs vs cats classification task (adding only a new layer trained from scratch) and keeping old layers frozen.
With custom model we train all the layers from scratch.

After training it is possible to export the model in pure tensorflow format, you could also easily add a CoreML conversion.

#### Training examples

Input images are automatically resized and processed depending on the model architecture.

Run script from *keras* directory.

Mobilenet alpha 1.0 (full size):

```
python3 ./tools/train.py
--datasetTrainDir ../,,/kaggle_dataset/trainval/train
--datasetValDir ../../kaggle_dataset/trainval/val
--configFile ./config/config_mobilenet_full.cfg
--modelOutputPath ./export/mobilenet_full_fn.h5
--tfModelOutputDir ./export/mobilenet_full_fn_TF
```

Mobilenet alpha 0.25 (mini):

```
python3 ./tools/train.py
--datasetTrainDir ../,,/kaggle_dataset/trainval/train
--datasetValDir ../../kaggle_dataset/trainval/val
--configFile ./config/config_mobilenet_mini.cfg
--modelOutputPath ./export/mobilenet_025_128_fn.h5
--tfModelOutputDir ./export/mobilenet_025_128_fn_TF
```

Custom model:

```
python3 ./tools/train.py
--datasetTrainDir ../,,/kaggle_dataset/trainval/train
--datasetValDir ../../kaggle_dataset/trainval/val
--configFile ./config/config_custom_model.cfg
--modelOutputPath ./export/custom_model.h5
--tfModelOutputDir ./export/custom_model_TF
```

Export directory will contains this models:
- Keras format (single h5 file with training stuff plus weights).
- TensorFlow files in dedicated directory.


### [Optional] Run predictions on test directory

This script runs predictions on dogs vs cats test images and exports results in a CSV format ready for kaggle submission.
Please notice that while the competition is over, you can still evaluate your model through [challenge page](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/leaderboard).

#### Test examples

Run script from *tensorflow* directory.

Mobilenet alpha 1.0 (full size):

```
python3 ./tools/test_dir_for_export.py
--datasetTestDir ../../kaggle_dataset/test
--modelPath ./export/mobilenet_full_fn.h5
--kaggleExportFile ./export/kaggle_mobilenet_full.csv
```

Mobilenet alpha 0.25 (mini):

```
python3 ./tools/test_dir_for_export.py
--datasetTestDir ../../kaggle_dataset/test
--modelPath ./export/mobilenet_025_128_fn.h5
--kaggleExportFile ./export/kaggle_mobilenet_mini.csv
```

### [Optional] Run predictions on test directory with generator

TODO
# Tensorflow Dogs Vs Cats example

Example usage of tensorflow to classify dogs and cats images (from Kaggle challenge [Dogs Vs Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)).

To use Nasnet model you need at least Tensorflow 1.4.

To convert SqueezeNet model from caffe format you need python 2.7, other scripts are tested on python 3.

## Models

### MobileNet

Started from Mobilenet pretrained on ImageNet, trained from scratch only the last convolution layer to classify the target classes. [[Paper]](https://arxiv.org/abs/1704.04861) [[Official TF slim models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).
First of all you need to put the pretrained graph.pb in baseModels/mobilenet directory, see [MobileNet README](./baseModels/mobilenet/README.md) for a step by step tutorial or download from [here](https://drive.google.com/open?id=19WOmkpEKZ0ohse5Jostre9m_7f-tUKTj) if you are lazy.

**Image format:** RGB.

**Image preprocessing:** transform from [0, 255] uint8 range to [-1.0, 1.0] float (Inception preprocessing).

#### Kaggle Results

Saved best epoch on validation accuracy.

| Model | Input Size | HyperParameters | Validation accuracy | Kaggle LogLoss|
|-------|------------|-----------------|---------------|------------------|
| MobileNet 1.0  | 224x224 | 20 epochs, LR Fixed 0.001, SGD optimizer | 0.9854 | 0.07604 |
| MobileNet 1.0  | 224x224 | 20 epochs, LR Exponential Decay start from 0.001, SGD optimizer | 0.9852 | 0.07771 |
| MobileNet 1.0  | 224x224 | 30 epochs, LR Start from 0.001, ADAM optimizer | 0.9896 | 0.07090 |

### NasNet

Started from Nasnet pretrained on ImageNet, trained from scratch only the last fully connected layer to classify the target classes. [[Paper]](https://arxiv.org/abs/1707.07012) [[Official TF slim models]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet).
First of all you need to put the pretrained graph.pb in baseModels/nasnet_mobile directory, see [NasNet README](./baseModels/nasnet/README.md) for a step by step tutorial or download from [here](https://drive.google.com/open?id=19WOmkpEKZ0ohse5Jostre9m_7f-tUKTj) if you are lazy.

**Image format:** RGB.

**Image preprocessing:** transform from [0, 255] uint8 range to [-1.0, 1.0] float (Inception preprocessing).

#### Kaggle Results

Saved best epoch on validation accuracy.

| Model | Input Size | HyperParameters | Validation accuracy | Kaggle LogLoss|
|-------|------------|-----------------|---------------|------------------|
| NasNet Mobile  | 224x224 | 30 epochs, LR Start from 0.001, ADAM optimizer | 0.9916 | 0.05663 |

### SqueezeNet

Started from SqueezeNet pretrained on ImageNet, trained from scratch only the last convolution layer to classify the target classes. [Paper](https://arxiv.org/abs/1602.07360) [[Official Caffe models]](https://github.com/DeepScale/SqueezeNet).
First of all you need to put the pretrained graph.pb in baseModels/squeezenet_v1.1 directory, see [SqueezeNet README](./baseModels/squeezenet_v1.1/README.md) for a step by step tutorial (caffe to TF conversion included) or download from [here](https://drive.google.com/open?id=19WOmkpEKZ0ohse5Jostre9m_7f-tUKTj) if you are lazy.

**Image format:** BGR (base model is trained with caffe)

**Image preprocessing:** input image in range [0.0, 255.0] float, then subtract the imagenet mean image (VGG preprocessing).

#### Kaggle Results

Saved best epoch on validation accuracy.

| Model | Input Size | HyperParameters | Validation accuracy | Kaggle LogLoss|
|-------|------------|-----------------|---------------|------------------|
| SqueezeNet v1.1  | 227x227 | 30 epochs, LR Start from 0.001, ADAM optimizer | 0.9792 | 0.12585 |

### Custom Model

Started from a custom model trained from scratch, the architecture inspired by VGG.

**Image format:** RGB.

**Image preprocessing:** input image in range [0.0, 255.0] float, then subtract the imagenet mean image (VGG preprocessing).

#### Kaggle Results

Saved best epoch on validation accuracy.

| Model | Input Size | HyperParameters | Validation accuracy | Kaggle LogLoss|
|-------|------------|-----------------|---------------|------------------|
| Custom  | 112x112 | 30 epochs, LR Start from 0.001, ADAM optimizer | 0.9160 | 0.48337 |



## Training instructions

Required steps:
- Create dataset tfrec train and validation files for fast training
- Fine tune training of the pretrained model on dogs vs cats classification task 

Utilities:
- Load any graph to tensorboard
- Run predictions on test subset and export results for kaggle competition

The directory *config* contains both dataset and training parameters for each model. You only need to pass a different config to the scripts to change model and hyperparameters.

The script calls examples considers that *../../kaggle_dataset* (kaggle_dataset in repo root) contains the extracted dogs vs cats kaggle dataset and contains this subdirectories:
- train: full training images
- test: test images for the competition

### Create dataset

Each model input size requires a different dataset, for simplicity we don't resize images after tfrec reload during training.

#### TFRec creation examples

Run script from *tensorflow* directory.
The script get as argument the full dataset train directory and automatically performs validation split.

SqueezeNet with input images 227x227x3:
```
python3 ./tools/createDatasetTFRec.py
--imagesDir ../../kaggle_dataset/train
--configFile ./config/squeezenet_v1.1_ADAM.cfg
--outputDir ../../tfrec_227
```

MobileNet with input images 224x224x3 (same dataset is valid also for NASNet mobile):
```
python3 ./tools/createDatasetTFRec.py
--imagesDir ../../kaggle_dataset/train
--configFile ./config/mobilenet_1.0_ADAM.cfg
--outputDir ../../tfrec_224
```

Output directory will contains:
- train.tfrec: actual training samples
- val.tfrec: validation samples
- metadata.json: metadata about dataset creation to reload data

### Training

We create a new branch in the architecture for dogs vs cats classification task (adding only a new layer trained from scratch) keeping old layers frozen, so we can start from the frozen graph.
After final freeze graph of the trained model, we discard the 1000 classes specific branch.

If you would like to continue training of existing weights, you need to load checkpoint instead of frozen graph.

The training script supports tensorboard for validation accuracy trend analysis over time and saves best epoch model on validation accuracy (checkpoint to continue training and frozen graph for deploy).

#### Training examples

Run script from *tensorflow* directory.

SqueezeNet v1.1 with ADAM optimizer:

```
python3 ./tools/train.py
--configFile ./config/squeezenet_v1.1_ADAM.cfg
--baseModelDir ./baseModels/squeezenet_v1.1
--datasetDir /home/michele/DNN/dogs_vs_cats_playground/kaggle_dataset/tfrec_227
--tensorboardDir ./tensorboard/squeezenet_ADAM/
--modelOutputDir ./trainedModels/squeezenet_ADAM
--checkpointOutputDir ./trainedModels/squeezenet_ADAM
--useGpu 0
```

NASNet Mobile with ADAM optimizer:

```
python3 ./tools/train.py
--configFile ./config/nasnet_mobile_ADAM.cfg
--baseModelDir ./baseModels/nasnet_mobile
--datasetDir /home/michele/DNN/dogs_vs_cats_playground/kaggle_dataset/tfrec_224
--tensorboardDir ./tensorboard/nasnet_mobile_ADAM/
--modelOutputDir ./trainedModels/nasnet_mobile_ADAM
--checkpointOutputDir ./trainedModels/nasnet_mobile_ADAM
--useGpu 0
```

Custom model with ADAM optimizer

```
python3 ./tools/train.py
--configFile ./config/custom_ADAM.cfg
--datasetDir /home/michele/DNN/dogs_vs_cats_playground/kaggle_dataset/tfrec_112
--tensorboardDir ./tensorboard/custom_ADAM/
--modelOutputDir ./trainedModels/custom_ADAM
--checkpointOutputDir ./trainedModels/custom_ADAM
--useGpu 0
```

### [Optional] Run predictions on test directory

This script runs predictions on dogs vs cats test images and exports results in a CSV format ready for kaggle submission.
Please notice that while the competition is over, you can still evaluate your model through [challenge page](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/leaderboard).

#### Test examples

Run script from *tensorflow* directory.

SqueezeNet v1.1 with ADAM optimizer:

```
python3 ./tools/testDirForExport.py
--configFile ./config/squeezenet_v1.1_ADAM.cfg
--modelPath ./trainedModels/squeezenet_ADAM/graph.pb
--datasetTestDir ../../kaggle_dataset/test
--kaggleExportFile ./export/kaggle_squeezenet_v1.1_ADAM.csv
```

NASNet Mobile with ADAM optimizer:

```
python3 ./tools/testDirForExport.py
--configFile ./config/nasnet_mobile_ADAM.cfg
--modelPath ./trainedModels/nasnet_mobile_ADAM/graph.pb
--datasetTestDir ../../kaggle_dataset/test
--kaggleExportFile ./export/kaggle_nasnet_mobile_ADAM.csv
```

### [Optional] Load graph into tensorboard

This script is useful to visualize the architecture, it works with pb files (frozen graph with weights or only graph definition).
It is useful to see layer names, choose which layers retrains, see layers input sizes, etc...

#### Load graph example

Run script from *tensorflow* directory.

```
python3 ./tools/loadGraphToTensorboard.py
--modelPath ./baseModels/nasnet_mobile/graph.pb
--tensorboardOutputDir ./tensorboard/nasnet_mobile_base_graph
```

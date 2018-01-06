# Turicreate Dogs Vs Cats example

Example usage of [turicreate by Apple](https://github.com/apple/turicreate) to classify dogs and cats images (from Kaggle challenge [Dogs Vs Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)).

Useful links:
- [Install guide from main repo README](https://github.com/apple/turicreate/blob/master/README.md)
- [Full turicreate API guide](https://apple.github.io/turicreate/docs/api/generated/turicreate)

I have succesfully run turicreate on this configuration:
- Ubuntu 16.04
- CUDA 8.0
- CuDNN 6.0

Tested version 4.0 with:
- python2.7
- mxnet-cu80: 0.11.0
- coremltools: 0.6.3
- numpy: 1.13.3

## Models

Turicreate [image classifier](https://apple.github.io/turicreate/docs/api/generated/turicreate.image_classifier.create.html#turicreate.image_classifier.create) supports only squeezenet 1.1 and resnet-50, it is not possible to define custom architectures.
The provided image classifier adds a logistic regression layer after the last base model layer.

Turicreate downloads the pretrained models from apple repository at the first usage.

### SqueezeNet

The model has 227x227x3 input size, but every resizing and preprocessing operations is automatically performed by turicreate-

#### Kaggle Results

| Model | HyperParameters | Best Validation accuracy | Final validation accuracy | Kaggle LogLoss|
|-------|---------|--------|---------------|------------------|
| SqueezeNet v1.1  | Max iterations 5000 | 0.960123 | 0.954755 | 0.18915 |

Optimal solution found at iteration 4555.

### Resnet-50

Cannot run, see "Known problems".

## Training instructions

Required steps:
- Create dataset in SFrame format
- Fine tune training of the pretrained model on dogs vs cats classification task 

Utilities:
- Run predictions on test subset and export results for kaggle competition

The directory *config* contains training parameters for each model.
You only need to pass a different config to the training script to change model and hyperparameters 
(very small options available with turicreate).

The script calls examples considers that *../../kaggle_dataset* (kaggle_dataset in repo root) contains the extracted 
dogs vs cats kaggle dataset and contains these subdirectories:
- train: full training images
- test: test images for the competition

### Create dataset

It is sufficient to create a single SFrame dataset, it is compatible with all models.

Run script from *turicreate* directory.
The script gets as argument the full dataset train directory.
```
python2 ./tools/createDataset.py
--imagesDir ../../kaggle_dataset/train
--outputDatasetFile ../../kaggle_dataset/sframe/dogs_vs_cats.sframe
```

Output file will contain the dataset in SFrame format, each image is saved in tc.Image format at full size.

### Fine tuning training

#### Training examples

Run script from *turicreate* directory.

SqueezeNet v1.1:

```
python2 ./tools/train.py
--datasetFile ../../kaggle_dataset/sframe/dogs_vs_cats.sframe
--configFile ./config/squeezenet_v1.1.cfg
--modelOutputDir ./export
```

Output directory will contain model in turicreate and coreml format.

### [Optional] Run predictions on test directory

This script runs predictions on dogs vs cats test images and exports results in a CSV format ready for kaggle submission.
Please notice that while the competition is over, you can still evaluate your model through [challenge page](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/leaderboard).

#### Test examples

Run script from *turicreate* directory.

SqueezeNet v1.1:

```
python2 ./tools/testDirForExport.py
--datasetTestDir ../../kaggle_dataset/test
--modelPath ./export/squeezenet_v1.1
--kaggleExportFile ./export/kaggle_squeezenet.csv
```

## Knows problems

#### Batch size not settable, possible out of memory

Turicreate doesn't permit to set batch size through image classifier API and uses 512 by default.

If you encounter "out of memory" error like me (see this github issues [137](https://github.com/apple/turicreate/issues/137) and [26](https://github.com/apple/turicreate/issues/26)) 
you need to change the default value in extract_features function in python turicreate image_feature_extractor.py source file (e.g. in *<anaconda_env>/lib/python2.7/site-packages/turicreate/toolkits/_image_feature_extractor.py*).

A batch size of 128 should be sufficient, you don't need to recompile the whole turicreate.

#### Cannot train resnet-50

Using the same script working for squeezenet, with my configuration I can't train resnet-50 model.

After the resize phase, with any batch size, the system stucks on best convolution algorithm search.

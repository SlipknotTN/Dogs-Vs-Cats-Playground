# Dogs Vs Cats playground

In this repo you will find different methods to solve the kaggle competition playground ["Dogs Vs Cats"](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition),
an image classification binary task.
My purpose is to show how to implement simple solutions with different python library publicily available.
It is not the scope of my work to benchmark the different methods in terms of accuracy or speed.

I will use small models suitable for a CUDA equipped notebook, you will not need a powerful computer with 4x Titan X to train the models.

My setup is a notebook with:
- CPU i7-6700HQ
- GPU Nvidia 970m
- RAM 16 GB
- Ubuntu 16.04

<p align="center"><img src="https://tbolttimes.com/wp-content/uploads/2016/03/dogs-and-cats.jpg" alt="Dogs Vs Cats"></p>

## Dataset

- Download the dataset from [Kaggle site](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
- Extract train and test directories in ./kaggle_dataset
- Create dog and cat subdirectories in train directory and move all images in the correct subdirectories
(image file names start with the class name)
- Further processing of the dataset specific for the frameworks are described in the relative inner sections

Kaggle dataset directory should look this:

```
kaggle_dataset
│
└───train
│   │
│   └───cat
│   │    │   cat.0.jpg
│   │    │   cat.1.jpg
│   │    │   ...
│   │
│   └───dog
│        │   dog.0.jpg
│        │   dog.1.jpg
│        │   ...
│
└───test
    │   1.jpg
    │   2.jpg
    │   ...
```

## Implementations roadmap

Each subproject is standalone and you can refer to subsequent links:

- [ ] XGBoost: [tutorial](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
- [ ] Deep Forest: [example](https://www.kaggle.com/demoon/the-nature-conservancy-fisheries-monitoring/deep-forest-1-4-public-log-loss-with-1-5-data)
- Deep Learning
    - [x] [Keras](/deep_learning/keras): pretrained MobileNet, custom model defined with Sequential API
    - [x] [Tensorflow](/deep_learning/tensorflow): pretrained MobileNet, pretrained SqueezeNet, pretrained NasNet, custom model
    - [x] [TuriCreate](/deep_learning/turicreate): pretrained SqueezeNet
    - [x] [PyTorch](/deep_learning/pytorch): pretrained SqueezeNet, pretrained ResNet50

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

![Dogs Vs Cats](https://kaggle2.blob.core.windows.net/competitions/kaggle/5441/logos/front_page.png)

## Dataset

TODO: How to download and prepare the dataset, each method can needs different processing (e.g. keras needs the trainval split).

## Implementantions roadmap

Each subproject is standalone and you can refer to subsequent links:

- [] XGBoost: [tutorial](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
- [] Deep Forest: [example](https://www.kaggle.com/demoon/the-nature-conservancy-fisheries-monitoring/deep-forest-1-4-public-log-loss-with-1-5-data)
- Deep Learning
    - [x] [Keras](/deep_learning/keras): pretrained MobileNet, custom model defined with Sequential API
    - [] [Tensorflow](/deep_learning/tensorflow): pretrained MobileNet, pretrained SqueezeNet, pretrained NasNet, custom model
    - [] PyTorch

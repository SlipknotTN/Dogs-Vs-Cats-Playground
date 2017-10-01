# Keras Dogs Vs Cats example

Example usage of keras to classify dogs and cats images (from Kaggle challenge [Dogs Vs Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)).

### Model

Mobilenet pretrained on ImageNet, trained from scratch only the last convolution layer to classify the target classes. [[Paper]](https://arxiv.org/abs/1704.04861) [[Official TF slim models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

**Image format:** RGB.

**Image preprocessing:** transform from [0, 255] uint8 range to [-1.0, 1.0] float (Inception preprocessing).

### Kaggle Results

| Model | Input Size | HyperParameters | Validation accuracy | Kaggle LogLoss|
|-------|------------|-----------------|---------------|------------------|
| MobileNet 1.0  | 224x224 | 10 epochs, LR 0.001, Momentum 0.9, SGD optimizer | 0.9795 | 0.08296 |
| MobileNet 0.25 | 128x128 | 10 epochs, LR 0.001, Momentum 0.9, SGD optimizer | 0.9049 | 0.24903 |

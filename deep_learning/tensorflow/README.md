# Tensorflow Dogs Vs Cats example

Example usage of tensorflow to classify dogs and cats images (from Kaggle challenge [Dogs Vs Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)).

To use Nasnet model you need at least Tensorflow 1.4.

## Models

### MobileNet

Start from Mobilenet pretrained on ImageNet, trained from scratch only the last convolution layer to classify the target classes. [[Paper]](https://arxiv.org/abs/1704.04861) [[Official TF slim models]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).
First of all you need to put the pretrained graph.pb in baseModels/mobilenet directory, see [MobileNet README](./baseModels/mobilenet/README.md).

**Image format:** RGB.

**Image preprocessing:** transform from [0, 255] uint8 range to [-1.0, 1.0] float (Inception preprocessing).

### Kaggle Results

Saved best epoch on validation accuracy.

| Model | Input Size | HyperParameters | Validation accuracy | Kaggle LogLoss|
|-------|------------|-----------------|---------------|------------------|
| MobileNet 1.0  | 224x224 | 20 epochs, LR Fixed 0.001, SGD optimizer | 0.9854 | 0.07604 |
| MobileNet 1.0  | 224x224 | 20 epochs, LR Exponential Decay start from 0.001, SGD optimizer | 0.9852 | 0.07771 |
| MobileNet 1.0  | 224x224 | 20 epochs, LR Start from 0.001, ADAM optimizer | 0.9896 | 0.07090 |
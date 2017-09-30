TODO: Add links to challenge, paper and tf slim mobilenet page.

# Keras Dogs Vs Cats example

Example usage of keras to classify dogs and cats images (Kaggle challenge).

### Model

Mobilenet pretrained on ImageNet, trained from scratch only the last convolution layer to classify the target classes.

Image format: RGB.

Image preprocessing: normalization in range [-1.0, 1.0] (Inception preprocessing).

### Kaggle Results

| Model | Input Size | HyperParameters | Validation accuracy | Kaggle LogLoss|
|-------|------------|-----------------|---------------|------------------|
| MobileNet 1.0 | 224x224 | 10 epochs, LR 0.001, Momentum 0.9, SGD optimizer | |0.08470      |
| MobileNet 0.25 | 128x128 | 10 epochs, LR 0.001, Momentum 0.9, SGD optimizer | |0.25757     |
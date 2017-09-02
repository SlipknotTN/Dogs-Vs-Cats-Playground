import argparse

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator


def preprocessing(image):
    """
    MobileNet preprocessing
    """
    image /= 255.0
    image -= 0.5
    image *= 2.0
    return image


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datasetDir", required=True, type=str, help="Dataset directory")
    args = parser.parse_args()
    return args


def main():

    # TODO: See this https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975 for reference

    args = doParsing()
    print(args)

    # Load MobileNet Full, with output shape of (None, 7, 7, 1024)
    baseModel = MobileNet(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False,
                          weights='imagenet', input_tensor=None, pooling=None)

    fineTunedModel = Sequential()
    fineTunedModel.add(baseModel)

    # Global average output shape (None, 1, 1, 1024).
    # Global pooling with AveragePooling2D to have a 4D Tensor and apply Conv2D.
    fineTunedModel.add(AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', data_format=None))

    # Convolution layer like TF MobileNet with 2 classes, output shape (None, 1, 1, 2)
    fineTunedModel.add(Conv2D(filters=2, kernel_size=(1,1), activation=None))

    # Image Generator, MobileNet needs [-1.0, 1.0] range
    trainImageGenerator = ImageDataGenerator(preprocessing_function=preprocessing, horizontal_flip=True)
    valImageGenerator = ImageDataGenerator(preprocessing_function=preprocessing)


if __name__ == '__main__':
    main()

import argparse

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, Activation
from keras.layers.pooling import AveragePooling2D
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# TODO: Create a config files for hyperparameters

def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Keras training script")
    parser.add_argument("--datasetTrainDir", required=True, type=str, help="Dataset train directory")
    parser.add_argument("--datasetValDir", required=True, type=str, help="Dataset validation directory")
    parser.add_argument("--modelOutputPath", required=False, type=str, default="./export/mobilenet_fn.h5",
                        help="Filepath where to save the model")
    parser.add_argument("--batchSize", required=False, type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", required=False, type=int, default=50, help="Number of training epochs")
    args = parser.parse_args()
    return args


def main():

    # See this example https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975 for reference
    # and this blog post https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

    args = doParsing()
    print(args)

    # Load MobileNet Full, with output shape of (None, 7, 7, 1024)
    baseModel = MobileNet(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False,
                          weights='imagenet', input_tensor=None, pooling=None)

    fineTunedModel = Sequential()

    fineTunedModel.add(baseModel)

    # Global average output shape (None, 1, 1, 1024).
    # Global pooling with AveragePooling2D to have a 4D Tensor and apply Conv2D.
    fineTunedModel.add(AveragePooling2D(pool_size=(baseModel.output_shape[1], baseModel.output_shape[2]),
                                        strides=(1, 1), padding='valid', name="global_pooling"))

    fineTunedModel.add(Dropout(rate=0.5))

    # Convolution layer that acts like fully connected, with 2 classes, output shape (None, 1, 1, 2)
    fineTunedModel.add(Conv2D(filters=2, kernel_size=(1, 1), name="fc_conv"))

    # Reshape to (None, 2) to match the one hot encoding target and final softmax
    fineTunedModel.add(Flatten())

    # Final sofmax for deploy stage
    fineTunedModel.add(Activation('softmax'))

    # Freeze the base model layers, train only the last convolution
    for layer in fineTunedModel.layers[0].layers:
        layer.trainable = False

    # Train as categorical crossentropy (works also for numclasses > 2)
    fineTunedModel.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                           metrics=['categorical_accuracy'])

    # Image Generator, MobileNet needs [-1.0, 1.0] range (Inception like preprocessing)
    trainImageGenerator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)
    valImageGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)

    trainGenerator = trainImageGenerator.flow_from_directory(
        args.datasetTrainDir,
        # height, width
        target_size=(224, 224),
        batch_size=args.batchSize,
        class_mode='categorical',
        shuffle=True)

    valGenerator = valImageGenerator.flow_from_directory(
        args.datasetValDir,
        # height, width
        target_size=(224, 224),
        batch_size=args.batchSize,
        class_mode='categorical',
        shuffle=False)

    # fine-tune the model
    fineTunedModel.fit_generator(
        trainGenerator,
        steps_per_epoch=trainGenerator.samples//trainGenerator.batch_size,
        epochs=args.epochs,
        validation_data=valGenerator,
        validation_steps=valGenerator.samples//valGenerator.batch_size)

    print("Training finished")

    # Save model and state
    fineTunedModel.save(args.modelOutputPath)

    print("Model saved to " + args.modelOutputPath)


if __name__ == '__main__':
    main()

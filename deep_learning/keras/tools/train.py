import _init_paths
import argparse

from tfutils.export import exportModelToTF
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
                        help="Filepath where to save the keras model")
    parser.add_argument("--tfModelOutputDir", required=False, type=str, default=None,
                        help="Optional directory where to export model in TF format (checkpoint + graph)")
    parser.add_argument("--inputSize", required=False, type=int, default=224, help="Square size of model input")
    parser.add_argument("--mobilenetAlpha", required=False, type=float, default=1.0, help="MobileNet alpha value")
    parser.add_argument("--batchSize", required=False, type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", required=False, type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()
    return args


def main():

    # See this example https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975 for reference
    # and this blog post https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

    args = doParsing()
    print(args)

    # Image Generator, MobileNet needs [-1.0, 1.0] range (Inception like preprocessing)
    trainImageGenerator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)
    valImageGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)

    trainGenerator = trainImageGenerator.flow_from_directory(
        args.datasetTrainDir,
        # height, width
        target_size=(args.inputSize, args.inputSize),
        batch_size=args.batchSize,
        class_mode='categorical',
        shuffle=True)

    valGenerator = valImageGenerator.flow_from_directory(
        args.datasetValDir,
        # height, width
        target_size=(args.inputSize, args.inputSize),
        batch_size=args.batchSize,
        class_mode='categorical',
        shuffle=False)

    # Load MobileNet Full, with output shape of (None, 7, 7, 1024)
    baseModel = MobileNet(input_shape=(args.inputSize, args.inputSize, 3), alpha=args.mobilenetAlpha,
                          depth_multiplier=1, dropout=1e-3, include_top=False,
                          weights='imagenet', input_tensor=None, pooling=None)

    fineTunedModel = Sequential()

    fineTunedModel.add(baseModel)

    # Global average output shape (None, 1, 1, 1024).
    # Global pooling with AveragePooling2D to have a 4D Tensor and apply Conv2D.
    fineTunedModel.add(AveragePooling2D(pool_size=(baseModel.output_shape[1], baseModel.output_shape[2]),
                                        strides=(1, 1), padding='valid', name="global_pooling"))

    fineTunedModel.add(Dropout(rate=0.5))

    # Convolution layer that acts like fully connected, with 2 classes, output shape (None, 1, 1, 2)
    fineTunedModel.add(Conv2D(filters=trainGenerator.num_class, kernel_size=(1, 1), name="fc_conv"))

    # Reshape to (None, 2) to match the one hot encoding target and final softmax
    fineTunedModel.add(Flatten())

    # Final sofmax for deploy stage
    fineTunedModel.add(Activation('softmax', name="softmax"))

    # Freeze the base model layers, train only the last convolution
    for layer in fineTunedModel.layers[0].layers:
        layer.trainable = False

    # Train as categorical crossentropy (works also for numclasses > 2)
    fineTunedModel.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                           metrics=['categorical_accuracy'])

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

    # Export model to TF format
    if args.tfModelOutputDir is not None:
        exportModelToTF(args.tfModelOutputDir)


if __name__ == '__main__':
    main()

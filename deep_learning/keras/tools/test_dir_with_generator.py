import argparse

from keras.models import load_model
from keras.applications import mobilenet
from keras.preprocessing.image import ImageDataGenerator


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Keras test script")
    parser.add_argument("--datasetTestDir", required=True, type=str,
                        help="Dataset test directory "
                             "(please notice that you need a dummy subdir also without labels data)")
    parser.add_argument("--modelPath", required=False, type=str, default="./export/mobilenet_fn.h5",
                        help="Filepath where to save the model")
    args = parser.parse_args()
    return args


def main():
    """
    Example of predict_generator usage for images without labels
    """
    args = doParsing()
    print(args)

    model = load_model(args.modelPath, custom_objects={
                       'relu6': mobilenet.relu6,
                       'DepthwiseConv2D': mobilenet.DepthwiseConv2D})

    print("Loaded model from " + args.modelPath)

    print(model.summary())

    testImageGenerator = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)

    testGenerator = testImageGenerator.flow_from_directory(
        args.datasetTestDir,
        # height, width
        target_size=(224, 224),
        batch_size=50,
        class_mode=None,
        shuffle=False)

    # List of #image ndarrays with shape #num_classes, each ndarray contains classes probabilities
    results = model.predict_generator(generator=testGenerator,
                                      steps=testGenerator.samples // testGenerator.batch_size +
                                            testGenerator.samples % testGenerator.batch_size)

    # If need it read results here, but please notice that you no references to filenames

    print("Test finished")


if __name__ == '__main__':
    main()

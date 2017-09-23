import argparse
import glob
import os

import numpy as np
from scipy.misc import imread, imresize

from keras.models import load_model
from keras.applications import mobilenet
from keras.preprocessing.image import ImageDataGenerator

classes = ['cat', 'dog']

def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Keras test script")
    parser.add_argument("--datasetTestDir", required=True, type=str, help="Dataset test directory")
    parser.add_argument("--modelPath", required=False, type=str, default="./export/mobilenet_fn.h5",
                        help="Filepath where to save the model")
    args = parser.parse_args()
    return args


def main():

    args = doParsing()
    print(args)

    model = load_model(args.modelPath, custom_objects={
                       'relu6': mobilenet.relu6,
                       'DepthwiseConv2D': mobilenet.DepthwiseConv2D})

    print("Loaded model from " + args.modelPath)

    print(model.summary())

    # Dogs and cats test dataset has 12500 samples

    # TODO: Create a custom generator, this one works only with labeled data, keep this as addition test with single image

    # testImageGenerator = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)

    # TODO: Check last batch extraction if not divisible by batch_size

    # testGenerator = testImageGenerator.flow_from_directory(
    #     args.datasetTestDir,
    #     # height, width
    #     target_size=(224, 224),
    #     batch_size=50,
    #     shuffle=False)
    #
    # results = model.predict_generator(generator=testGenerator, steps=testGenerator.samples//testGenerator.batch_size)

    for file in sorted(glob.glob(args.datasetTestDir + "/*.jpg")):

        # One by one image prediction

        # Image processing (resize and inception like preprocessing to have [-1.0, 1.0] input range)
        image = imread(file)
        image = imresize(image, size=model.input_shape[1:3])
        image = image.astype(np.float32)
        processedImage = mobilenet.preprocess_input(image)

        # Add 1st dimension for image index in batch
        processedImage = np.expand_dims(processedImage, axis=0)

        result = model.predict_classes(x=processedImage, batch_size=1, verbose=False)
        print(os.path.basename(file) + " -> " + classes[result[0]])

    print("Test finished")

    # TODO: Add to libs CSV kaggle build

if __name__ == '__main__':
    main()

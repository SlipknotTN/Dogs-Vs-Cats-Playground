import _init_paths
import argparse
import glob
import os

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from tqdm import tqdm

from config.ConfigParams import ConfigParams
from image.ImageUtils import ImageUtils
from model.TensorflowModel import TensorflowModel


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="TF Profiling script")
    parser.add_argument("--datasetTestDir", required=True, type=str, help="Dataset test directory")
    parser.add_argument("--configFile", required=True, type=str, help="Model config file")
    parser.add_argument("--modelPath", required=False, type=str, default="./export/graph.pb",
                        help="Filepath with trained model")
    parser.add_argument("--numImages", required=False, type=int, default=100,
                        help="Number of images to use during profiling")
    args = parser.parse_args()
    return args


def main():
    """
    Script to export TF profiling results,
    tutorial from https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d
    You will need Google Chrome to read exported tracing.
    """
    args = doParsing()
    print(args)

    # Load config (it includes preprocessing type)
    config = ConfigParams(args.configFile)

    # Load model
    model = TensorflowModel(args.modelPath)

    print("Loaded model from " + args.modelPath)

    inputPlaceholder = model.getGraph().get_tensor_by_name(config.inputName + ":0")
    outputTensor = model.getGraph().get_tensor_by_name(config.outputName + ":0")

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # One by one image prediction forcing CPU usage
    with model.getSession() as sess:

        with tf.device("/gpu:0"):

            for ix, file in tqdm(enumerate(sorted(glob.glob(args.datasetTestDir + "/*.jpg"))[:args.numImages])):

                image = ImageUtils.loadImage(file)
                # Resize image and preprocess (inception or vgg preprocessing based on config)
                processedImage = ImageUtils.preprocessing(image=image, width=config.inputSize, height=config.inputSize,
                                                          preprocessingType=config.preprocessType,
                                                          meanRGB=config.meanRGB)

                # Convert colorspace
                processedImage = ImageUtils.convertImageFormat(processedImage, format=config.inputFormat)

                # Add 1st dimension for image index in batch
                processedImage = np.expand_dims(processedImage, axis=0)

                # Get and print TOP1 class
                result = sess.run(outputTensor, feed_dict={inputPlaceholder: processedImage},
                                  options=options, run_metadata=run_metadata)
                print(os.path.basename(file) + " -> " + str(np.argmax(result[0])))

                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # Create chrome tracing which you can load with chrome://tracing inside Chrome browser
                os.makedirs(os.path.join('./profiling/', config.architecture), exist_ok=True)
                with open(os.path.join('profiling', config.architecture, 'timeline_step_%d.json' % ix), 'w') as f:
                    f.write(chrome_trace)

    print("Test finished")


if __name__ == '__main__':
    main()

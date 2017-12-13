import argparse
import json
import os
import tensorflow as tf

from config.dataset.DatasetParams import DatasetParams
from trainUtils.ArgsValidator import ArgsValidator
from data.DatasetWriterFactory import DatasetWriterFactory


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Script for classification fine tuning')
    parser.add_argument('--imagesDir', required=True, help='Folder containing images.')
    parser.add_argument('--datasetConfigFile', required=True, type=str, help='Config file for dataset creation')
    parser.add_argument('--outputDir', required=True, help='TFRecords destination directory, use a clean directory')
    parser.add_argument('--debugRead', action='store_true',
                        help="Enable imshow debug for images read from stored records")
    parser.add_argument('--proposalsDir', required=False,
                        help='Folder containing objectness proposals, needed for detection')
    parser.add_argument('--groundThruthBoxesDir', required=False, help='Folder containing detection ground thruth, '
                                                                       'needed for detection')
    parser.add_argument('--trainFile', required=False, default=None,
                        help='File containing training examples filenames (without extensions)')
    parser.add_argument('--valFile', required=False, default=None,
                        help='File containing validation examples filenames (without extensions)')
    return parser.parse_args()


def main():

    args = do_parsing()
    print(args)

    # Read dataset configuration
    datasetParams = DatasetParams().initFromConfigFile(args.datasetConfigFile).setImagesDir(args.imagesDir)

    # Get dataset writer with training and validation splits
    dataset = DatasetWriterFactory.createDatasetWriter(datasetParams=datasetParams, scriptArgs=args)

    if os.path.exists(args.outputDir) is False:
        os.makedirs(args.outputDir)

    trainingOutputFile = os.path.join(args.outputDir, "data_train.tfrecords")
    validationOutputFile = os.path.join(args.outputDir, "data_val.tfrecords")
    jsonFilePath = os.path.join(args.outputDir, "metadata.json")

    # Export Train Samples
    with tf.python_io.TFRecordWriter(trainingOutputFile) as tfrecWriter:
        print("TRAINING")
        dataset.saveTFExamplesTraining(datasetParams=datasetParams, writer=tfrecWriter)
        print("Saving file...")

    # Export Validation Samples
    with tf.python_io.TFRecordWriter(validationOutputFile) as tfrecWriter:
        print("VALIDATION")
        dataset.saveTFExamplesValidation(datasetParams=datasetParams, writer=tfrecWriter)
        print("Saving file...")

    # Export metadata to JSON, we need to set the additional parameters retrieved from actual images directory
    trainingSamplesNumber = dataset.getTrainingSamplesNumber()
    validationSamplesNumber = dataset.getValidationSamplesNumber()
    datasetParams = datasetParams.setNumClasses(dataset.numClasses) \
                        .setTraining(trainingSamplesNumber, os.path.basename(trainingOutputFile)) \
                        .setValidation(validationSamplesNumber, os.path.basename(validationOutputFile))

    with open(jsonFilePath, 'w') as jsonOutFile:
        json.dump(datasetParams, jsonOutFile, default=lambda o: o.__dict__, indent=4)

    print ("Dataset successfully created in " + args.outputDir)

if __name__ == '__main__':
    main()
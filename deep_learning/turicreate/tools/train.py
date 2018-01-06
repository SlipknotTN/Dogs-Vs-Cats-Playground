import _init_paths
import argparse
import os

import turicreate as tc

from config.ConfigParams import ConfigParams
from constants.Constants import Constants as const


def doParsing():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Script to train turicreate image classifier')
    parser.add_argument('--datasetFile', required=True, help="SFrame file path")
    parser.add_argument('--configFile', required=True, help="Config file path")
    parser.add_argument('--modelOutputDir', required=False, default="./trainedModels",
                        help="Output directory for trained model pickle")
    return parser.parse_args()


def main():

    args = doParsing()
    print(args)

    # Read training configuration (config file is in common for dataset creation and training hyperparameters)
    configParams = ConfigParams(args.configFile)

    # Load data
    data = tc.SFrame(args.datasetFile)

    # Create and train model
    model = tc.image_classifier.create(data, model=configParams.architecture, max_iterations=configParams.iterations,
                                       target=const.DatasetFeatures.targets, verbose=True)

    # Save model
    model.save(os.path.join(args.modelOutputDir, configParams.architecture))
    print("Model saved")

    # Export to Core ML
    model.export_coreml(os.path.join(args.modelOutputDir, configParams.architecture + '.mlmodel'))
    print("CoreML model exported")


if __name__ == "__main__":
    main()

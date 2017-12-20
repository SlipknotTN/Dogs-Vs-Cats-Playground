import _init_paths
import argparse
import os

from config.ConfigParams import ConfigParams
from trainUtils.trainDevice import selectTrainDevice
from model.TensorflowModel import TensorflowModel


def doParsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Tensorflow image classification fine tuning')
    parser.add_argument('--datasetDir', required=True, default=None, help='Dataset directory')
    parser.add_argument('--modelDir', type=str, required=True, help='Base model folder')
    parser.add_argument('--checkpointOutputDir', required=False, default="./export",
                        help='Output folder that will contains checkpoints')
    parser.add_argument('--modelOutputDir', required=False, default="./export",
                        help='Output folder that will contains final trained model graph.pb')
    parser.add_argument('--configFile', required=True, help='Config File for training')
    parser.add_argument('--tensorboardDir', required=False, default=None, help="TensorBoard directory")
    parser.add_argument('--useGpu', type=str, required=False, default=None, help='Gpu to use for the training ')
    return parser.parse_args()


def main():

    args = doParsing()
    print(args)

    # Read training configuration (config file is in common for dataset creation and training hyperparameters)
    trainingParams = ConfigParams(args.configFile)

    # Select train device
    trainDevice = selectTrainDevice(args.useGpu)

    # TODO: Load DataProvider

    # Load base model graph
    baseTFModel = TensorflowModel(os.path.join(args.modelDir, "graph.pb"))

    # TODO: Append classifier for fine tuning training (use ModelFactory)


    # TODO: Run training loading images

    # TODO: Save fine tuned model


if __name__ == '__main__':

    main()

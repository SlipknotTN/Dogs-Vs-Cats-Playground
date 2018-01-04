import argparse
import os

import turicreate as tc


def doParsing():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Script to train turicreate image classifier')
    parser.add_argument('--datasetFile', required=True, help="SFrame file path")
    parser.add_argument('--modelArchitecture', required=True, help="SFrame file path")
    parser.add_argument('--modelOutputDir', required=False, default="./trainedModels",
                        help="Output directory for trained model pickle")
    return parser.parse_args()


def main():

    args = doParsing()
    print(args)

    # Load data
    data = tc.SFrame(args.datasetFile)

    # Create a model
    model = tc.image_classifier.create(data, model=args.modelArchitecture, max_iterations=1000, target="targets", verbose=True)

    # Make predictions
    #predictions = model.predict(data)

    # Save model
    model.save(os.path.join(args.modelOutputDir, args.modelArchitecture))
    print("Model saved")

    # Export to Core ML
    model.export_coreml(os.path.join(args.modelOutputDir, args.modelArchitecture + '.mlmodel'))
    print("CoreML model exported")


if __name__ == "__main__":
    main()

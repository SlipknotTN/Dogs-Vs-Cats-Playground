import _init_paths
import argparse
import glob
import os

import numpy as np
import turicreate as tc

from kaggle.export import exportResults


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Turicreate test script, single images analysis")
    parser.add_argument("--datasetTestDir", required=True, type=str, help="Dataset test directory")
    parser.add_argument("--modelPath", required=True, type=str, help="Filepath with trained model (tc save dir)")
    parser.add_argument("--kaggleExportFile", required=False, type=str, default=None,
                        help="CSV file in kaggle format for challenge upload")
    args = parser.parse_args()
    return args


def main():
    """
    Script to export results for Kaggle, Images are read one by one
    """
    args = doParsing()
    print(args)

    # Load model
    model = tc.load_model(args.modelPath)

    print("Loaded model from " + args.modelPath)

    # Dogs and cats test dataset has 12500 samples

    results = []

    # One by one image prediction
    for imageFile in sorted(glob.glob(args.datasetTestDir + "/*.jpg")):

        # Load image as tc Image, no explicit resize to model input size
        image = tc.Image(imageFile)

        # Single image SFrame compatible with model utility functions
        sframe = tc.SFrame(data={"features": [image]})

        # Get and print TOP1
        probabilities = model.predict(sframe, output_type="probability_vector")
        print(os.path.basename(imageFile) + " -> " + model.classes[int(np.argmax(probabilities.to_numpy()[0]))])

        # Get and save dog probability
        results.append((os.path.basename(imageFile)[:os.path.basename(imageFile).rfind('.')], probabilities[0][model.classes.index("dog")]))

    print("Test finished")

    if args.kaggleExportFile is not None:
        exportResults(results, args.kaggleExportFile)


if __name__ == '__main__':
    main()

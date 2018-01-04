import os
import argparse
import glob

import turicreate as tc

from tqdm import tqdm


def doParsing():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Script to create SFrame dataset for turicreate image classifier')
    parser.add_argument('--imagesDir', required=True, help="Input images directory containing classes subdirectories")
    parser.add_argument('--outputDatasetFile', required=True, help="Output SFrame file path")
    return parser.parse_args()


def main():

    args = doParsing()
    print(args)

    sframeDict = {"features": [], "targets": []}

    classesSubDirs = next(os.walk(args.imagesDir))[1]

    for classSubDir in tqdm(classesSubDirs):

        for imageFile in tqdm(glob.glob(os.path.join(args.imagesDir, classSubDir) + "/*.jpg")):

            image = tc.Image(imageFile)
            imageResized = tc.image_analysis.resize(image, width=227, height=227, channels=3)
            sframeDict["features"].append(imageResized)
            sframeDict["targets"].append(classSubDir)

    print("Saving SFrame...")

    datasetSFrame = tc.SFrame(data=sframeDict)
    datasetSFrame.save(filename=args.outputDatasetFile)

    print("SFrame saved in " + args.outputDatasetFile)


if __name__ == "__main__":
    main()
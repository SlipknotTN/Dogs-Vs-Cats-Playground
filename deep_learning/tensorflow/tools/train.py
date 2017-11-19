import argparse
import sys


def doParsing():

    parser = argparse.ArgumentParser(description="Tensorflow image classification fine tuning script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument(name="--piero", required=True)
    return parser


def main():


    # TODO: Read args
    args = doParsing()

    # TODO: Read config

    # TODO: Load pretrained model

    # TODO: Run training loading images

    # TODO: Save fine tuned model

    print ("Ciao")


if __name__ == '__main__':

    main()

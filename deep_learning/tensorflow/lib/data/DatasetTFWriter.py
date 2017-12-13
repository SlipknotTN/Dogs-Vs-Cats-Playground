import cv2
import numpy as np
import tensorflow as tf

import os

from tqdm import tqdm

from data.DatasetWriter import DatasetWriter
from trainUtils.TrainConstants import TrainConstants as constants


class DatasetTFWriter(DatasetWriter):

    def __init__(self):
        DatasetWriter.__init__(self)

    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def bytes_list_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def saveTFExamplesTraining(self, datasetParams, writer):

        self.saveTFExamples(samplesList=self.trainingSamplesList, datasetParams=datasetParams, writer=writer,
                            subset=constants.Subsets.training)

    def saveTFExamplesValidation(self, datasetParams, writer):

        self.saveTFExamples(samplesList=self.validationSamplesList, datasetParams=datasetParams, writer=writer,
                            subset=constants.Subsets.validation)

    def saveTFExamples(self, samplesList, datasetParams, writer, subset):

        sess = None
        if datasetParams.imageEncoding == constants.FileFormats.jpeg:
            sess = tf.Session()

        # Placeholder and encode_jpeg op, it doesn't support float32
        imageInput = tf.placeholder(dtype=tf.uint8)
        imageEncodingOp = tf.image.encode_jpeg(imageInput)

        for sample in tqdm(samplesList, desc=subset + " dataset creation progress"):

            filename = sample[0]
            label = sample[1]

            # Load image in HWC-RGB uint8 format
            originalImage = ImageUtils.loadImage(filename)
            imageRaw = ImageUtils.loadAndResizeSquash(originalImage,
                                                      size=(datasetParams.imageWidth, datasetParams.imageHeight))

            # Encoding of sample image
            image = None
            if datasetParams.imageEncoding is None:
                image = imageRaw.tostring()
            elif datasetParams.imageEncoding == constants.FileFormats.jpeg:
                image = sess.run(imageEncodingOp, feed_dict={imageInput: imageRaw.astype(np.uint8)})

            sample = tf.train.Example(
                # Example contains a Features proto object
                features=tf.train.Features(
                    # Features contains a map of string to Feature proto objects
                    feature={
                        constants.DatasetFeatures.label: self.int64_feature(label),
                        # Uncompressed bytes save, uint8 serialized to a single utf-8 string
                        constants.DatasetFeatures.image: self.bytes_feature(tf.compat.as_bytes(image)),
                    }))

            # use the proto object to serialize the example to a string
            serialized = sample.SerializeToString()
            # write the serialized object to disk
            writer.write(serialized)

        if sess is None:
            sess.close()

        return

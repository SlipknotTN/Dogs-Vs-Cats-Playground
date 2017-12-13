from .DatasetTFWriter import DatasetTFWriter


class DatasetWriterFactory(object):

    @classmethod
    def createDatasetWriter(cls, datasetParams, scriptArgs):

        # Init Dataset TF Writer
        dataset = DatasetTFWriter()
        dataset.setTrainValSamplesList(imagesDir=scriptArgs.imagesDir,
                                       validationPercentage=datasetParams.validationPercentage)

        return dataset

    @classmethod
    def setDetectionSamples(cls, dataset, datasetParams, scriptArgs):

        # Single directory, train validation split with percentage
        if datasetParams.trainValFiles is False:
            dataset.setTrainValSamplesListDetection(datasetParams=datasetParams, scriptArgs=scriptArgs)

        # Single directory, train and validation separated files with samples list
        else:
            dataset.setVOCTrainSamplesListDetection(datasetParams=datasetParams, scriptArgs=scriptArgs)
            dataset.setVOCValSamplesListDetection(datasetParams=datasetParams, scriptArgs=scriptArgs)

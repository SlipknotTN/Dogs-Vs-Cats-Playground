class Constants(object):

    class ConfigSection:

        hyperparameters = "HYPERPARAMETERS"
        model = "MODEL"
        datasetParameters = "DATASET_PARAMETERS"

    class DatasetParams:

        validationPercentage = "validationPercentage"
        imageEncoding = "imageEncoding"

    class DatasetFeatures:

        features = "features"
        targets = "targets"

    class TrainConfig:
        numEpochs = 'numEpochs'

class Constants(object):

    class ConfigSection:

        hyperparameters = "HYPERPARAMETERS"
        model = "MODEL"
        datasetParameters = "DATASET_PARAMETERS"

    class DatasetParams:

        validationPercentage = "validationPercentage"
        imageEncoding = "imageEncoding"

    class PreprocessingType:

        vgg = "vgg"
        inception = "inception"

    class FileFormats:

        jpeg = "jpeg"
        png = "png"

    class Subsets:

        training = "Training"
        validation = "Validation"

    class DatasetFeatures:

        label = "label"
        image = "image"

import configparser
from constants.Constants import Constants as const

class ConfigParams(object):

    def __init__(self, file):

        config = configparser.ConfigParser()
        config.read_file(open(file))

        # Model
        self.architecture = config.get(const.ConfigSection.model, "architecture")
        # Valid only for mobilenet
        if self.architecture == "mobilenet":
            self.mobilenetAlpha = config.getfloat(const.ConfigSection.model, "mobilenetAlpha", fallback=1.0)
        self.inputSize = config.getint(const.ConfigSection.model, "inputSize", fallback=224)
        self.inputChannels = config.getint(const.ConfigSection.model, "inputChannels", fallback=3)
        self.preprocessType = config.get(const.ConfigSection.model, "preprocessType", fallback="dummy")

        # HyperParameters
        self.epochs = config.getint(const.ConfigSection.hyperparameters, "epochs")
        self.batchSize = config.getint(const.ConfigSection.hyperparameters, "batchSize")
        self.patience = config.getint(const.ConfigSection.hyperparameters, "patience")
        self.learningRate = config.getfloat(const.ConfigSection.hyperparameters, "learningRate")
        self.optimizer = config.get(const.ConfigSection.hyperparameters, "optimizer")
        if self.optimizer != "SGD":
            raise Exception("Only SGD optimizer supported")

        #Dataset creation params (image size = model size for simplicity)
        self.validationPercentage = config.getint(const.ConfigSection.datasetParameters,
                                                  const.DatasetParams.validationPercentage)
        self.imageEncoding = config.get(const.ConfigSection.datasetParameters, const.DatasetParams.imageEncoding,
                                        fallback=None)

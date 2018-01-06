import ConfigParser

from constants.Constants import Constants as const


class ConfigParams(object):

    def __init__(self, file):

        config = ConfigParser.ConfigParser()
        config.readfp(open(file))

        # Model
        self.architecture = config.get(const.ConfigSection.model, "architecture")

        # HyperParameters
        self.iterations = config.getint(const.ConfigSection.hyperparameters, "iterations")

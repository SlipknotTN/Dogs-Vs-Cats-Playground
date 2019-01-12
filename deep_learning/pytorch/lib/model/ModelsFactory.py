from .SqNet import SqNet
from .ResNet50 import ResNet50


class ModelsFactory(object):

    @classmethod
    def create(cls, config, num_classes):

        if config.architecture == "sqnet":

            # Transfer learning
            return SqNet(num_classes)

        elif config.architecture == "resnet50":

            # Transfer learning
            return ResNet50(num_classes)

        else:

            raise Exception("Model architecture " + config.architecture + " not supported")
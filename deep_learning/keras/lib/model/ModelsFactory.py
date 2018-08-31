from .Model import Model


class ModelsFactory(object):

    @classmethod
    def create(cls, config, numClasses):

        inputShape = (config.inputSize, config.inputSize, config.inputChannels)

        if config.architecture == "mobilenet":
            # Transfer learning
            return Model.mobilenet(inputShape=inputShape, numClasses=numClasses, alpha=config.mobilenetAlpha,
                                   retrainAll=False)

        elif config.architecture == "custom":
            # Train from scratch
            return Model.custom(inputShape=inputShape, numClasses=numClasses)

        else:

            raise Exception("Model architecture " + config.architecture + " not supported")
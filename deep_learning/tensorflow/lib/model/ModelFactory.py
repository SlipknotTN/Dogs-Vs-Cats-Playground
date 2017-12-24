from .MobileNet import MobileNet


class ModelFactory(object):

    @classmethod
    def create(cls, config, tfmodel, dataProvider, trainDevice):

        # Choose model network and build trainable layers
        if config.architecture.lower() == "squeezenet":
            raise Exception('Architecture ' + config.model + 'not supported')
            # TODO: Support SqueezeNet
            # return SqueezeNet(model=tfmodel, trainingParams=config,
            #                                dataProvider=dataProvider,
            #                                trainDevice=trainDevice)
        elif config.architecture.lower() == "mobilenet":
            return MobileNet(configParams=config, model=tfmodel, dataProvider=dataProvider, trainDevice=trainDevice)
        else:
            raise Exception('Architecture ' + config.model + 'not supported')

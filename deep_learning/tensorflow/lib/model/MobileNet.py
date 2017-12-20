import tensorflow as tf

from .ClassificationModel import ClassificationModel


class MobileNet(ClassificationModel):

    def __init__(self, model, trainingParams, dataProvider, trainDevice):

        ClassificationModel.__init__(self)

        self.setPlaceholders(trainDevice=trainDevice, model=model, datasetParams=dataProvider.datasetParams)

        # Custom fine tuning definition, train only the last classifier
        with tf.device(trainDevice):
            # Loaded model is freezed in test mode
            self.inputLayerTrainedFromScratch = model.getGraph().get_tensor_by_name(
                'MobilenetV1/Logits/Dropout_1b/Identity:0')
            self.layersTrainedFromScratchNames = trainingParams.layersTrainedFromScratch

            # Tensorflow ops for layers representation
            self.lastConv = tf.layers.conv2d(
                self.inputLayerTrainedFromScratch,
                filters=dataProvider.datasetParams.numClasses,
                kernel_size=[1,1],
                strides=(1, 1),
                padding='same',
                data_format='channels_last',
                dilation_rate=(1, 1),
                activation=None,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name="Conv2d_1c_1x1_fn",
                reuse=None
            )

            # Reshape to 2D array (batch x numClasses), otherwise convolution output is 4D
            self.logits = tf.reshape(self.lastConv, [-1, dataProvider.datasetParams.numClasses])

        self.setTrainableVariables(trainDevice=trainDevice, trainingParams=trainingParams)
        self.defineTrainingOperations(dataProvider, model, trainDevice, trainingParams)

    def defineTrainingOperations(self, dataProvider, model, trainDevice, trainingParams):

        self.setDeployOutputs(trainDevice, trainingParams=trainingParams)

        with tf.device('/cpu:0'):
            self.globalStep = tf.Variable(0, trainable=False)

        self.setCostFunction(trainDevice)

        with tf.device(trainDevice):
            self.learningRate = LearningRateFactory.createLearningRate(optimizerParams=trainingParams.optimizer,
                                                                    trainBatches=dataProvider.getTrainBatchesNumber(),
                                                                    globalStepVar=self.globalStep)

        with tf.device('/cpu:0'):
            self.lrateSummary = tf.summary.scalar("learning rate", self.learningRate)

        with tf.device(trainDevice):
            self.optimizer = TrainOptimizer.createOptimizer(learningRate=self.learningRate,
                                                            optimizerParams=trainingParams.optimizer)
            self.optimizer = self.optimizer.minimize(self.cost, global_step=self.globalStep, var_list=self.var_list)

        with tf.device('/cpu:0'):
            self.correctPred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))

        with tf.device(trainDevice):
            self.accuracy = tf.reduce_mean(tf.cast(self.correctPred, tf.float32), name='accuracy')
            self.scalarInputForSummary = tf.placeholder(dtype=tf.float32, name="scalar_input_summary")

        with tf.device('/cpu:0'):
            self.costSummary = tf.summary.scalar("Training loss", self.scalarInputForSummary)
            self.accuracySummary = tf.summary.scalar("validation accuracy", self.scalarInputForSummary)


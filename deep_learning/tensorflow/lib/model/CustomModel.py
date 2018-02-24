import tensorflow as tf

from .ClassificationModel import ClassificationModel


class CustomModel(ClassificationModel):

    def __init__(self, configParams, model, dataProvider, trainDevice):

        ClassificationModel.__init__(self, configParams, model, dataProvider, trainDevice)

        with tf.device(trainDevice):

            # Tensorflow Placeholders
            self.x = tf.placeholder(dtype=tf.float32,
                                    shape=[None, configParams.inputSize, configParams.inputSize, configParams.inputChannels],
                                    name=configParams.inputName)
            # Ground truth placeholder (one-hot encoding)
            self.y = tf.placeholder(dtype=tf.int32, shape=[None, self.dataProvider.datasetMetadata.numClasses], name="y")

            # Layers trained from scratch, architecture inspired by VGG
            conv1 = tf.layers.conv2d(
                self.x,
                filters=64,
                kernel_size=[3,3],
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="conv1"
            )

            conv2 = tf.layers.conv2d(
                conv1,
                filters=64,
                kernel_size=[3,3],
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="conv2"
            )

            pool1 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), name="pool1")

            conv3 = tf.layers.conv2d(
                pool1,
                filters=128,
                kernel_size=[3,3],
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="conv3"
            )

            conv4 = tf.layers.conv2d(
                conv3,
                filters=128,
                kernel_size=[3,3],
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="conv4"
            )

            pool2 = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=(2,2), name="pool2")

            conv5 = tf.layers.conv2d(
                pool2,
                filters=256,
                kernel_size=[3, 3],
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="conv5"
            )

            conv6 = tf.layers.conv2d(
                conv5,
                filters=256,
                kernel_size=[3, 3],
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="conv6"
            )

            pool3 = tf.layers.max_pooling2d(conv6, pool_size=(2, 2), strides=(2, 2), name="pool3")


            conv7 = tf.layers.conv2d(
                pool3,
                filters=256,
                kernel_size=[3, 3],
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="conv7"
            )

            conv8 = tf.layers.conv2d(
                conv7,
                filters=256,
                kernel_size=[3, 3],
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="conv8"
            )

            pool4 = tf.layers.max_pooling2d(conv8, pool_size=(2, 2), strides=(2, 2), name="pool4")

            reshape = tf.layers.Flatten()(pool4)

            fc1 = tf.layers.dense(
                inputs=reshape,
                units=4096,
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="fc1"
            )

            fc2 = tf.layers.dense(
                inputs=fc1,
                units=4096,
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="fc2"
            )

            fc3 = tf.layers.dense(
                inputs=fc2,
                units= self.dataProvider.datasetMetadata.numClasses,
                use_bias=True,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="fc3"
            )

            # Reshape to 2D array (batchSize x numClasses), otherwise convolution output is 4D
            self.logits = tf.reshape(fc3, [-1, self.dataProvider.datasetMetadata.numClasses])

        # Set all variables to varList (no need to filter
        self.varList = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.defineTrainingOperations()

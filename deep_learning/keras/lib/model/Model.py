from keras.models import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dropout, Dense
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D
from keras.applications.mobilenet import MobileNet


class Model(object):

    @classmethod
    def custom(cls, inputShape, numClasses):
        """
        Sequential API model definition example using common layers
        (this architecture doesn't perform well on this task, use only as example to define your own model)
        """

        model = Sequential()

        # Convolutions with stride 2 are like convolution + pooling
        # 2x smaller on width and height -> 112x112x64
        model.add(Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=64,
                         input_shape=inputShape,
                         padding='same', activation='relu', name="conv1"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

        # 2x smaller on width and height -> 56x56x128
        model.add(
            Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=128, padding='same', activation='relu', name="conv2"))

        # 2x smaller on width and height -> 28x28x256
        model.add(
            Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=256, padding='same', activation='relu', name="conv3"))

        # 2x smaller on width and height -> 14x14x512
        model.add(
            Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=512, padding='same', activation='relu', name="conv4"))

        # 2x smaller on width and height -> 7x7x1024
        model.add(
            Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=1024, padding='same', activation='relu', name="conv5"))

        # Final classifier

        # Double Fully connected
        model.add(Dropout(rate=0.5))
        model.add(Flatten())
        model.add(Dense(units=1024))
        model.add(Dense(units=numClasses))

        # Conv 1x1 + GAP
        #model.add(Dropout(rate=0.5))
        #model.add(Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=numClasses))
        #model.add(GlobalAveragePooling2D())

        model.add(Activation('softmax', name="softmax"))

        return model

    @classmethod
    def mobilenet(cls, inputShape, numClasses, alpha, retrainAll=False):

        # Load MobileNet Full, with output shape of (None, 7, 7, 1024), final classifier is excluded
        baseModel = MobileNet(input_shape=inputShape, alpha=alpha,
                              depth_multiplier=1, dropout=1e-3, include_top=False,
                              weights='imagenet', input_tensor=None, pooling=None)

        # Adding final classifier
        fineTunedModel = Sequential()

        fineTunedModel.add(baseModel)

        # Global average output shape (None, 1, 1, 1024).
        # Global pooling with AveragePooling2D to have a 4D Tensor and apply Conv2D.
        fineTunedModel.add(AveragePooling2D(pool_size=(baseModel.output_shape[1], baseModel.output_shape[2]),
                                            strides=(1, 1), padding='valid', name="global_pooling"))

        fineTunedModel.add(Dropout(rate=0.5))

        # Convolution layer that acts like fully connected, with 2 classes, output shape (None, 1, 1, 2)
        fineTunedModel.add(Conv2D(filters=numClasses, kernel_size=(1, 1), name="fc_conv"))

        # Reshape to (None, 2) to match the one hot encoding target and final softmax
        fineTunedModel.add(Flatten())

        # Final sofmax for deploy stage
        fineTunedModel.add(Activation('softmax', name="softmax"))

        # Freeze the base model layers, train only the last convolution
        if retrainAll is False:
            for layer in fineTunedModel.layers[0].layers:
                layer.trainable = False

        return fineTunedModel

import cv2
import numpy as np
from .ImageException import ImageException
from constants.Constants import Constants as const


class ImageUtils(object):
    """
    Image Utils for tensorflow pipeline, but in python (no TF operations)
    """
    @classmethod
    def loadImage(cls, imageFile):
        """
        Load image with OpenCV (forced to read as RGB)
        :param imageFile:
        :return: ndarray uint8 HWC-RGB
        """
        image = cv2.imread(imageFile, cv2.IMREAD_COLOR)
        # Check image corruption
        if image is None:
            raise ImageException("Error reading image: " + imageFile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.uint8)

    @classmethod
    def loadAndResizeSquash(cls, image, size):
        """
        Load and resize image (squash)
        :param image: ndarray uint8 HWC-RGB
        :param size: tuple (width, height)
        :return: Resized image in ndarray format HWC-RGB uint8
        """
        if image.shape[0:2] == size:
            rsz = image
        else:
            # Always inter_area cv2 resize
            rsz = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        # Force uint8 conversion (however it is the default format of imread)
        return rsz.astype(np.uint8)

    @classmethod
    def preprocessing(cls, image, width, height, preprocessingType):
        """
        :param image ndarray RGB-HWC uint8 in range [0, 255]
        """

        # We use the passed width and height (not forced to use imageParams width and height)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)

        image = image.astype(np.float32)

        if preprocessingType == const.PreprocessingType.vgg:
            raise Exception("Preprocessing type " + preprocessingType + " not supported")
            # TODO: Add support for VGG preprocessing (useful for SqueezeNet)
            #image = cls.preprocessVgg(image, imageParams.mean)
        elif preprocessingType == const.PreprocessingType.inception:
            image = cls.preprocessInception(image)
        else:
            raise Exception("Preprocessing type " + preprocessingType + " not supported")

        return image

    @classmethod
    def preprocessVgg(cls, image, mean):
        """
        VGG Processing (equal to AlexNet, SqueezeNet, GoogleNet, ...)
        :param image ndarray RGB-HWC float32 in range [0.0, 255.0]
        :param mean Mean values RGB (ndarray of 3 elements)
        :return: processed image in format float32 HWC-RGB Tensor in range [0.0, 255.0]
        minus mean (we can obtain negative values)
        """
        assert (len(image.shape) == 3), "VGG preprocessing supports only 3 channels images"

        # Subtract mean
        image[:,:,0] -= mean[0]
        image[:,:,1] -= mean[1]
        image[:,:,2] -= mean[2]
        return image

    @classmethod
    def preprocessInception(cls, image):
        """
        Inception and MobileNet preprocessing
        :param image ndarray RGB-HWC float32 in range [0.0, 255.0]
        :return: float32 HWC-RGB Tensor in range [-1.0, 1.0]
        """
        # Min max normalization forces values to min and max, we avoid this with following operations
        image *= 1.0/255.0
        image -= 0.5
        image *= 2.0
        return image


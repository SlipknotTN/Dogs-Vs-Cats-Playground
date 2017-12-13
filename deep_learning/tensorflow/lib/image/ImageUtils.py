import cv2
import numpy as np


class ImageUtils(object):
    """
    Image Utils for tensorflow pipeline, but in python (no TF operations)
    """
    @classmethod
    def loadImage(cls, imageFile):
        """
        Load image with OpenCV
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


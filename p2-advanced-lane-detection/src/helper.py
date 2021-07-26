import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging

def getBasicLogger(logPath = None):
    """Gets a basic logger. In case of no log file path, this default logger
    prints the messages.
    """

    format = '%(asctime)-15s : %(levelname)s : %(filename)s : %(funcName)s ' + \
        ':: %(message)s'

    logging.basicConfig(filename=logPath, format=format, level=logging.INFO)

    return logging.getLogger('default')

def plotMultiple(images, gridShape, titles = None):
    """Helps plot multiple images based on the grid shape."""

    if len(gridShape) != 2 or (gridShape[0] * gridShape[1]) != len(images):
        ex = RuntimeError("Grid shape is invalid or doesn't match images count")

        raise ex

    if titles is not None and len(titles) != len(images):
        ex = RuntimeError("Titles ({0}) and images ({1}) count mismatch"\
            .format(len(titles), len(images)))

        raise ex

    if titles is None:
        titles = [str(i) for i in range(len(images))]

    fig, ax = plt.subplots(gridShape[0], gridShape[1])

    for i, (image, title) in enumerate(zip(images, titles)):

        cmap = None

        if len(image.shape) < 3 or image.shape[2] == 1:
            cmap = 'gray'

        if gridShape[0] == 1 or gridShape[1] == 1:
            ax[i].set_title(title)
            ax[i].imshow(image, cmap=cmap)
            continue

        r = i // gridShape[1]
        c = i - r * gridShape[1]

        ax[r][c].set_title(title)
        ax[r][c].imshow(image, cmap=cmap)

    plt.show()

def colorThreshold(hlsImage, thresh = (120, 255)):
    """Filters an image based on the S channel thresholds.

    Args:
        hlsImage : Image to perform thresholding upon in HLS format.
        thresh : The inclusive threshold range. Defaults to [120, 255].

    Returns:
        Binary mask.
    """

    sChannel = hlsImage[:, :, 2]

    colorThreshImg = np.zeros_like(sChannel)
    colorThreshImg[(sChannel >= thresh[0]) & (sChannel <= thresh[1])] = 1

    return colorThreshImg

def gradThreshold(hlsImage, magThresh = (50, 150), dirThresh = (0.6, 1.3)):
    """Filters an image based on the gradient magnitude and direction
        thresholds. A pixel is marked white if it's white in both the magnitude
        and direction masks.

    Args:
        hlsImage : Image to perform thresholding upon in HLS format.
        magThresh : The magnitude threshold range. Defaults to [50, 150].
        dirThresh : The direction threshold range. Defaults to [50, 150].

    Returns:
        Binary mask.
    """

    magThreshImg = gradMagThreshold(hlsImage, magThresh)
    dirThreshImg = gradDirThreshold(hlsImage, dirThresh)

    gradThreshImg = np.zeros_like(dirThreshImg)
    gradThreshImg[(magThreshImg == 1) & (dirThreshImg == 1)] = 1

    return gradThreshImg

def gradMagThreshold(hlsImage, thresh = (50, 150)):
    """Filters an image based on the gradient magnitude thresholds.

    Args:
        hlsImage : Image to perform thresholding upon in HLS format.
        thresh : The inclusive threshold range. Defaults to [50, 150].

    Returns:
        Binary mask.
    """

    lChannel = hlsImage[:, :, 1]

    sxImg = cv2.Sobel(lChannel, cv2.CV_64F, 1, 0)
    syImg = cv2.Sobel(lChannel, cv2.CV_64F, 0, 1)

    absSImg = np.sqrt(sxImg ** 2 + syImg ** 2)
    scaledSImg = (255. * absSImg / np.max(absSImg)).astype(np.uint8)

    magThreshImg = np.zeros_like(scaledSImg)
    magThreshImg[(scaledSImg >= thresh[0]) & \
        (scaledSImg <= thresh[1])] = 1

    return magThreshImg

def gradDirThreshold(hlsImage, thresh = (0.6, 1.3)):
    """Filters an image based on the gradient direction thresholds.

    Args:
        hlsImage : Image to perform thresholding upon in HLS format.
        thresh : The inclusive radian threshold range. Defaults to [0.6, 1.3].

    Returns:
        Binary mask.
    """

    lChannel = hlsImage[:, :, 1]

    sxImg = cv2.Sobel(lChannel, cv2.CV_64F, 1, 0)
    syImg = cv2.Sobel(lChannel, cv2.CV_64F, 0, 1)

    # absDirImg ends up being between 0 and pi/2 (1.57 rad).
    absDirImg = np.absolute(np.arctan2(syImg, np.absolute(sxImg)))

    dirThreshImg = np.zeros_like(absDirImg)
    dirThreshImg[(absDirImg >= thresh[0]) & (absDirImg <= thresh[1])] = 1

    return dirThreshImg


def logAndThrowError(logger, message):
    """Logs and raises a RunTime error with given message."""

    ex = RuntimeError(message)

    if logger is not None:
        logger.exception(ex)

    raise ex
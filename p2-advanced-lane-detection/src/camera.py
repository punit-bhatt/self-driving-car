import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

from helper import logAndThrowError

class Camera:

    def __init__(self, logger):
        """Initializes a new Camera instance."""

        self.logger = logger
        self.K = None
        self.distCoeffs = None
        self.R = None
        self.t = None

    def calibrateCamera(self,
                        chessboardImgDirPath,
                        gridPattern,
                        drawChessboard = False):
        """Calibrates the camera and returns the intrinsic and extrinsic
        parameters.

        Args:
            chessboardImgDirPath : Path of directory containing calibration
                images.
            gridPattern : Inner corner grid pattern.
            drawChessboard : Value indicating if chessboard with corners need
                to be plotted. Defaults to False.

        Returns:
            Camera matrix and distortion coefficients computed.

        """

        if not os.path.exists(chessboardImgDirPath):
            logAndThrowError(self.logger,
                             "Path '{0}' does not exist.".format( \
                             chessboardImgDirPath))

        chessboardImgPaths = [chessboardImgDirPath + imgName \
            for imgName in os.listdir(chessboardImgDirPath)]

        # Creating a list of the 3-D coordinates of the chessboard corners by
        # assuming the chessboard plane to be Z = 0.
        # Using mgrid to populate the x, y coordinates with appropriate values.
        coord3dObjPts = np.zeros((gridPattern[0] * gridPattern[1], 3), np.float32)
        grid = np.mgrid[:gridPattern[0], :gridPattern[1]]
        coord3dObjPts[:, :2] = grid.T.reshape(-1, 2)

        objPts = []
        imgPts = []

        for path in chessboardImgPaths:

            image = mpimg.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            retval, corners = cv2.findChessboardCorners(gray, gridPattern)

            self.logger.info('Found corners for {0}: {1}'.format(path, retval))

            if retval:

                objPts.append(coord3dObjPts)
                imgPts.append(corners)

                if drawChessboard:
                    imageWithCorners = cv2.drawChessboardCorners(image,
                                                                gridPattern,
                                                                corners,
                                                                retval)

                    plt.imshow(imageWithCorners)
                    plt.show()

        retval, K, distCoeffs, R, t = cv2.calibrateCamera(objPts,
                                                          imgPts,
                                                          gray.shape,
                                                          None,
                                                          None)
        self.logger.info('K = {0}'.format(K))
        self.logger.info('Distortion Coefficients = {0}'.format(distCoeffs))
        self.logger.info('R = {0}'.format(R))
        self.logger.info('t = {0}'.format(t))

        self.K = K
        self.distCoeffs = distCoeffs
        self.R = R
        self.t = t

    def undistort(self, image):
        """Returns an undistorted image."""

        if (self.K is None) or (self.distCoeffs is None):
            logAndThrowError(self.logger,
                             'K or distortion coefficients missing. ' + \
                             'Calibrate camera first!')

        return cv2.undistort(image, self.K, self.distCoeffs, None, self.K)

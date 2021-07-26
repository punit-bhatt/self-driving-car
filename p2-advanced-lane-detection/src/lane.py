import cv2
import matplotlib.pyplot as plt
import numpy as np


from helper import logAndThrowError, colorThreshold, gradThreshold

class Lane:

    def __init__(self,
                 logger,
                 nwindowHist = 9,
                 histMargin = 75,
                 histMinPix = 50,
                 roiMargin = 75,
                 priorWeight = 0.,
                 confidenceThresh = 0.1):
        """Initializes a new Lane instance.

        Args:
            logger : Logger instance.
            nwindowHist : Number of histogram windows. Defaults to 9.
            histMargin : Histogram x margin. Defaults to 75.
            histMinPix : Minimum pixels to shift histogram center. Defaults to
                50.
            roiMargin : ROI margin. Defaults to 75.
            priorWeight : Weight to be assigned to the previous curve while
                calculating current. Defaults to 0..
            confidenceThresh : Confidence threshold to use previous curve as
                reference. Defaults to 0.1.
        """

        self.logger = logger
        self.__roiMargin = roiMargin
        self.__nwindowHist = nwindowHist
        self.__histMargin = histMargin
        self.__histMinPix = histMinPix
        self.__priorWeight = priorWeight
        self.__confidenceThresh = confidenceThresh

        self.__warpMat = None
        self.__unwarpMat = None
        self.__confidence = 0.
        self.__leftLaneFit = None
        self.__rightLaneFit = None
        self.__radiusCurvature = None
        self.__frameIndex = -1

    def detectEdges(self, image):
        """Detects edges in an image using S (saturation) channel and gradient
        thresholding.

        Args:
            image : RGB image to detect edges for.

        Returns:
            Binary mask with edges.
        """

        hlsImage = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        colorThreshImg = colorThreshold(hlsImage, (100, 255))
        gradThreshImg = gradThreshold(hlsImage,
                                      magThresh=(50, 150),
                                      dirThresh=(0.7, 1.3))

        edges = np.zeros_like(colorThreshImg)
        edges[(colorThreshImg == 1) | (gradThreshImg == 1)] = 1

        return edges

    def perspectiveTransform(self, srcPts, dstPts):
        """Helps generate the warp and unwarp matrices based on reference source
        and destination points.

        Args:
            srcPts : Source reference points.
            dstPts : Corresponding destination points.
        """

        if len(srcPts) != len(dstPts):
            logAndThrowError(self.logger,
                             'Source and destination points count mismatch.')

        self.__warpMat = cv2.getPerspectiveTransform(srcPts, dstPts)
        self.__unwarpMat = cv2.getPerspectiveTransform(dstPts, srcPts)

    def findLane(self, image, drawOut = False):
        """Finds lane by either using a brute force histogram sliding window
        technique or using previous frame information on a warped image. These
        choices are made on the basis of confidence of the previous frame lane
        curves.

        Args:
            image : Image
            drawOut : Value indicating if ROI needs to be plotted.
                Defaults to False.

        Returns:
            Left lane curve coefficiensts.
            Right lane curve coefficients.
            Lane decision confidence.
            Out image with relevant plots.
        """

        if len(image.shape) != 3:
            logAndThrowError(self.logger, 'Pass an RGB image.')

        h = image.shape[0]
        w = image.shape[1]

        edges = self.detectEdges(image)
        warpedEdges = self.warpImage(edges)

        if self.__confidence < self.__confidenceThresh:
            laneLFit, laneRFit, confidence, radius, roiImg = \
                self.__findLaneByHistogramSlidingWindow__(warpedEdges, drawOut)
        else:
            laneLFit, laneRFit, confidence, radius, roiImg = \
                self.__findLaneByPreviousCurve__(warpedEdges, drawOut)

        self.__leftLaneFit = laneLFit
        self.__rightLaneFit = laneRFit
        self.__confidence = confidence
        self.__radiusCurvature = radius
        self.__frameIndex += 1

        self.logger.info(('Frame {0}: Confidence - {1}, LeftLane - {2},' + \
            'RightLane - {3}, RCurve - {4}.').format(self.__frameIndex,
                                                     confidence,
                                                     laneLFit,
                                                     laneRFit,
                                                     radius))

        outImg = None

        if drawOut:
            outImg, _, _ = self.__drawLane__(image,
                                             warpedEdges,
                                             roiImg)

        return laneLFit, laneRFit, confidence, outImg

    def warpImage(self, image):
        """Warps an image using the warp matrix.

        Args:
            image : Image to be warped.

        Returns:
            Warped image.
        """

        if self.__warpMat is None:
            logAndThrowError(self.logger, 'Warp matrix not generated yet!')

        imgshape = image.shape

        if len(imgshape) > 2:
            imgshape = imgshape[:2]

        return cv2.warpPerspective(image,
                                   self.__warpMat,
                                   imgshape[::-1],
                                   flags=cv2.INTER_LINEAR)

    def unwarpImage(self, warpedImg):
        """Unwarps an image using the unwarp matrix.

        Args:
            warpedImg : Warped image.

        Returns:
            Unwarped image.
        """

        if self.__unwarpMat is None:
            logAndThrowError(self.logger,
                             'Unwarp (Warp inv) matrix not generated yet!')

        imgshape = warpedImg.shape

        if len(imgshape) > 2:
            imgshape = imgshape[:2]

        return cv2.warpPerspective(warpedImg,
                                   self.__unwarpMat,
                                   imgshape[::-1],
                                   flags=cv2.INTER_LINEAR)

    def __findLaneByHistogramSlidingWindow__(self, warpedImg, drawOut = False):
        """Finds lane by using a brute force histogram sliding window technique.

        Args:
            image : Image
            drawOut : Value indicating if ROI needs to be plotted.
                Defaults to False.

        Returns:
            Left lane curve coefficiensts.
            Right lane curve coefficients.
            Lane decision confidence.
            Radius of curvature.
            Out image with relevant plots.
        """

        assert (len(warpedImg.shape) < 3)

        h = warpedImg.shape[0]
        w = warpedImg.shape[1]

        hMid = h // 2
        wMid = w // 2

        baseHist = np.sum(warpedImg[hMid :], axis = 0)
        baseLx = np.argmax(baseHist[:wMid])
        baseRx = np.argmax(baseHist[wMid:]) + wMid

        windowWidth = h // self.__nwindowHist
        nwindows = self.__nwindowHist

        if h % nwindows != 0:
            nwindows += 1

        currLx = baseLx
        currRx = baseRx

        nonzero = warpedImg.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        lInds = []
        rInds = []
        rectangles = []

        for i in range(nwindows):

            winHighY = h - i * windowWidth
            winLowY = winHighY - windowWidth if winHighY >= windowWidth else 0

            winLxLow = currLx - self.__histMargin
            winLxHigh = currLx + self.__histMargin

            winRxLow = currRx - self.__histMargin
            winRxHigh = currRx + self.__histMargin

            if drawOut:
                rectangles.append([(winLxLow, winLowY), (winLxHigh, winHighY)])
                rectangles.append([(winRxLow, winLowY), (winRxHigh, winHighY)])

            winLInds = ((nonzeroy >= winLowY) & (nonzeroy < winHighY) & \
                (nonzerox >= winLxLow) & (nonzerox < winLxHigh)).nonzero()[0]
            winRInds = ((nonzeroy >= winLowY) & (nonzeroy < winHighY) & \
                (nonzerox >= winRxLow) & (nonzerox < winRxHigh)).nonzero()[0]

            lInds.append(winLInds)
            rInds.append(winRInds)

            if len(winLInds) >= self.__histMinPix:
                currLx = int(np.mean(nonzerox[winLInds]))
            if len(winRInds) >= self.__histMinPix:
                currRx = int(np.mean(nonzerox[winRInds]))

        # log op times everywhere.

        lInds = np.concatenate(lInds)
        rInds = np.concatenate(rInds)

        lx = nonzerox[lInds]
        ly = nonzeroy[lInds]
        rx = nonzerox[rInds]
        ry = nonzeroy[rInds]

        # In our case, 'x'/rows represent Y and 'y'/columns represent X.
        laneLFit = np.polyfit(ly, lx, deg = 2)
        laneRFit = np.polyfit(ry, rx, deg = 2)

        # Calculating Radius of Curvature
        radius = self.__calculateRadiusCurvature__(h, w, laneLFit, laneRFit)

        outImg = None

        if drawOut:
            outImg = np.dstack((warpedImg, warpedImg, warpedImg))

            # All non-zero points considered lane
            outImg[ly, lx] = (255, 0, 0)
            outImg[ry, rx] = (0, 0, 255)

            # Window rectangles
            for rect in rectangles:
                cv2.rectangle(outImg, rect[0], rect[1], (0, 255, 0), 3)

            laneLines, _ = self.__getLaneMask__(h, w, laneLFit, laneRFit)
            outImg = cv2.addWeighted(outImg, 1.5, laneLines, 1, 0)

        # 2 * HistMargin pixels are covered in every row for left and right
        # lane. Thus, total pixel -> 2 * 2 * HistMargin * #rows
        totalRoiPixels = (4 * self.__histMargin) * h
        confidence = self.__calculateConfidence__(totalRoiPixels,
                                                  len(lx) + len(rx))

        return laneLFit, laneRFit, confidence, radius, outImg

    def __findLaneByPreviousCurve__(self, warpedImg, drawOut = False):
        """Finds lane by using previous frame lane curve to narrow the lane ROI.

        Args:
            image : Image
            drawOut : Value indicating if ROI needs to be plotted.
                Defaults to False.

        Returns:
            Left lane curve coefficiensts.
            Right lane curve coefficients.
            Lane decision confidence.
            Radius of curvature.
            Out image with relevant plots.
        """

        assert (len(warpedImg.shape) < 3)
        assert (self.__leftLaneFit is not None)
        assert (self.__rightLaneFit is not None)

        h = warpedImg.shape[0]
        w = warpedImg.shape[1]

        nonzero = warpedImg.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        lInds = []
        rInds = []

        prevLx = self.__leftLaneFit[0] * (nonzeroy ** 2) + \
            self.__leftLaneFit[1] * nonzeroy + self.__leftLaneFit[2]
        prevRx = self.__rightLaneFit[0] * (nonzeroy ** 2) + \
            self.__rightLaneFit[1] * nonzeroy + self.__rightLaneFit[2]

        lInds = ((nonzerox >= (prevLx - self.__roiMargin)) & \
            (nonzerox < (prevLx + self.__roiMargin))).nonzero()[0]
        rInds = ((nonzerox >= (prevRx - self.__roiMargin)) & \
            (nonzerox < (prevRx + self.__roiMargin))).nonzero()[0]

        lx = nonzerox[lInds]
        ly = nonzeroy[lInds]
        rx = nonzerox[rInds]
        ry = nonzeroy[rInds]

        # In our case, 'x'/rows represent Y and 'y'/columns represent X.
        laneLFit = np.polyfit(ly, lx, deg = 2)
        laneRFit = np.polyfit(ry, rx, deg = 2)

        laneLFit = tuple(map(lambda x, y: (1 - self.__priorWeight) * x + \
                             self.__priorWeight * y,
                             laneLFit,
                             self.__leftLaneFit))
        laneRFit = tuple(map(lambda x, y: (1 - self.__priorWeight) * x + \
                             self.__priorWeight * y,
                             laneRFit,
                             self.__rightLaneFit))

        radius = self.__calculateRadiusCurvature__(h, w, laneLFit, laneRFit)

        outImg = None

        if drawOut:
            outImg = np.dstack((warpedImg, warpedImg, warpedImg))
            winImg = np.zeros_like(outImg)

            # All non-zero points considered lane
            outImg[ly, lx] = (255, 0, 0)
            outImg[ry, rx] = (0, 0, 255)


            y = np.array(range(h))
            prevLaneL = self.__leftLaneFit[0] * (y ** 2) + \
                self.__leftLaneFit[1] * y + self.__leftLaneFit[2]
            prevLaneR = self.__rightLaneFit[0] * (y ** 2) + \
                self.__rightLaneFit[1] * y + self.__rightLaneFit[2]

            # To maintain continuity of contour points, either l or r points
            # need to be considered in reverse ordered.
            lRoiPts = np.vstack((np.dstack((prevLaneL - self.__roiMargin, y)),
                                 np.dstack((prevLaneL[::-1] + self.__roiMargin,
                                            y[::-1]))))
            lRoiPts = lRoiPts.reshape((-1, 2))
            rRoiPts = np.vstack((np.dstack((prevLaneR - self.__roiMargin, y)),
                                 np.dstack((prevLaneR[::-1] + self.__roiMargin,
                                            y[::-1]))))
            rRoiPts = rRoiPts.reshape((-1, 2))

            # Plotting ROI window
            cv2.fillPoly(winImg, np.int_([lRoiPts]), color =(0,205,0))
            cv2.fillPoly(winImg, np.int_([rRoiPts]), color =(0,205,0))

            laneLines, _ = self.__getLaneMask__(h, w, laneLFit, laneRFit)

            outImg = cv2.addWeighted(outImg, 1.5, winImg, 0.2, 0)
            outImg = cv2.addWeighted(outImg, 1.5, laneLines, 1, 0)

        # 2 * RoiMargin pixels are covered in every row for left and right lane.
        # Thus, total pixel -> 2 * 2 * roiMargin * #rows
        totalRoiPixels = (4 * self.__roiMargin) * h
        confidence = self.__calculateConfidence__(totalRoiPixels,
                                                  len(lx) + len(rx))

        return laneLFit, laneRFit, confidence, radius, outImg

    def __calculateConfidence__(self, totalRoiPixels, lanePixelsInRoi):
        """Calculates confidence as the ratio of area covered by lane pixels to
        that of the entire ROI.
        """

        return lanePixelsInRoi / totalRoiPixels

    def __getLaneMask__(self,
                        h,
                        w,
                        laneLFit,
                        laneRFit,
                        thickness = 3,
                        lColor = (255, 255, 0),
                        rColor = (255, 255, 0),
                        fill = False):
        """Generates a lane mask based on the curve coefficients passed.

        Args:
            h : Image height.
            w : Image width.
            laneLFit : Left lane curve coefficients.
            laneRFit : Right lane curve coefficients.
            thickness : Lane line thickness. Defaults to 3.
            lColor : Left lane color. Defaults to (255, 255, 0).
            rColor : Right lane color. Defaults to (255, 255, 0).
            fill : Value indicating if a fill mask needs to be created. Defaults
                to False.

        Returns:
            Lane mask with left and right in different colors.
            Lane area mask in a different color.
        """

        laneMask = np.zeros((h, w, 3), dtype=np.uint8)

        y = np.arange(h)
        lx = laneLFit[0] * (y ** 2) + laneLFit[1] * y + laneLFit[2]
        rx = laneRFit[0] * (y ** 2) + laneRFit[1] * y + laneRFit[2]

        # To handle sharp curves towards left or right. FOr such curves, some x
        # values would go below 0 or exceed the width.
        lxInRange = (lx >= 0) & (lx < w)
        rxInRange = (rx >= 0) & (rx < w)

        lPts = np.dstack((lx[lxInRange], y[lxInRange])).reshape(-1, 2)
        rPts = np.dstack((rx[rxInRange], y[rxInRange])).reshape(-1, 2)

        fillMask = None

        if fill:
            fillMask = np.zeros((h, w, 3), dtype=np.uint8)
            fillColor = (127, 255, 212)

            # To maintain continuity of contour points, either l or r points
            # need to be considered in reverse ordered.
            pts = np.vstack((lPts, rPts[::-1]))

            cv2.fillPoly(fillMask, np.int_([pts]), fillColor)

        cv2.polylines(laneMask, np.int_([lPts]), False, lColor, thickness)
        cv2.polylines(laneMask, np.int_([rPts]), False, rColor, thickness)

        return laneMask, fillMask

    def __calculateRadiusCurvature__(self, h, w, laneLFit, laneRFit):
        """Calculates the radius of curvature.

        Args:
            h : Image height.
            w : Image width.
            laneLFit : Left lane curve coefficients.
            laneRFit : Right lane curve coefficients.

        Returns:
            Average radius of curvature.
        """

        y = h - 1

        # y and x in meters per pixels. 600 based on the warped ROI points.
        my = 30 / h
        mx = 3.7 / 600

        y_eval = y * my

        mCoeff0 = mx / (my ** 2)
        mCoeff1 = mx / my

        leftCurve = ((1 + (2 * mCoeff0 * laneLFit[0] * y_eval + mCoeff1 * \
            laneLFit[1]) ** 2) ** 1.5) / abs(2 * mCoeff0 * laneLFit[0])
        rightCurve = ((1 + (2 * mCoeff0 * laneRFit[0] * y_eval + mCoeff1 * \
            laneRFit[1]) ** 2) ** 1.5) / abs(2 * mCoeff0 * laneRFit[0])

        return (leftCurve + rightCurve) / 2.

    def __drawLane__(self, image, warpedEdges, roiImg):
        """Draws unwarped lanes, relevant plots and information onto the
        original frame.

        Args:
            image : Original image
            warpedEdges : Warped edges image to be added to the output.
            roiImg : ROI image generated by findLane methods to be added to the
                output.

        Returns:
            Output image with lane lines marked and filled. Also, has the Radius
            of curvature mentioned and other relevant images plotted in picture.
        """

        if self.__leftLaneFit is None or self.__rightLaneFit is None:
            logAndThrowError(self.logger, 'Perform lane detection first!')

        h = image.shape[0]
        w = image.shape[1]

        warpedImg = self.warpImage(image)

        laneMask, fillMask = self.__getLaneMask__(h,
                                                  w,
                                                  self.__leftLaneFit,
                                                  self.__rightLaneFit,
                                                  thickness=30,
                                                  lColor=(255, 0, 0),
                                                  rColor=(255, 0, 0),
                                                  fill=True)
        unwarpedLaneMask = self.unwarpImage(laneMask)
        unwarpedFillMask = self.unwarpImage(fillMask)

        # Marking the lanes and the area between.
        outImg = cv2.addWeighted(image, 1, unwarpedFillMask, 0.3, 0)
        outImg = cv2.addWeighted(outImg, 0.9, unwarpedLaneMask, 1.5, 0)
        outWarpedImage = cv2.addWeighted(warpedImg, 0.9, laneMask, 1.5, 0)
        outWarpedImage = cv2.addWeighted(outWarpedImage, 1, fillMask, 0.3, 0)

        # Adding pict-in-pict for ROI, warped edges and warped out.
        resizeH = image.shape[0] * 20 // 100
        resizeW = image.shape[1] * 20 // 100
        resizedRoi = cv2.resize(roiImg,
                                (resizeW, resizeH),
                                interpolation = cv2.INTER_AREA)
        resizedWarped = cv2.resize(outWarpedImage,
                                    (resizeW, resizeH),
                                    interpolation = cv2.INTER_AREA)
        resizedEdges = cv2.resize(warpedEdges * 255.,
                                    (resizeW, resizeH),
                                    interpolation = cv2.INTER_AREA)
        resizedEdges = np.dstack((resizedEdges, resizedEdges, resizedEdges))

        # Adding the resized images in the top-right corner.
        outImg[:resizeH, w - resizeW : w, :] = resizedWarped
        outImg[:resizeH, w - 2 * resizeW : w - resizeW] = resizedRoi
        outImg[:resizeH, w - 3 * resizeW : w - 2 * resizeW] = resizedEdges

        # Adding radius of curvature text in top-right corner.
        text = 'R: {0}'.format(round(self.__radiusCurvature, 3))
        bottomLeft = (10, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontColor = (0, 0, 0)
        fontScale = 1.5
        lineType = 2

        cv2.putText(outImg,
                    text,
                    bottomLeft,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        return outImg, laneMask, fillMask

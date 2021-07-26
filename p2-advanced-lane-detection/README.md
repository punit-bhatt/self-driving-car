# **Advanced Lane Finding**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistort1]: ./output_images/cam_cal_undistort_1.png "Undistorted"
[undistort2]: ./output_images/cam_cal_undistort_3.png "Undistorted"
[hls_lab]: ./output_images/hls_lab2.png "hls_lab"
[Thresholded]: ./output_images/color_grad_thresholded.png "Thresholded"
[perspective_transform]: ./output_images/perspective_transform1.png "perspective_transform"
[perspective_transform2]: ./output_images/perspective_transform2.png "perspective_transform2"
[lane_roi1]: ./output_images/lane_roi_1.png "lane_roi1"
[lane_roi2]: ./output_images/lane_roi_2.png "lane_roi2"
[output]: ./output_images/output.png "output"

[//]: # (File References)

[camera.py]: ./src/camera.py "camera.py"
[helper.py]: ./src/helper.py "helper.py"
[lane.py]: ./src/lane.py "lane.py"
[log file]: ./src/log/log1.txt "log file"
[output video]: ./out_project_video.mp4 "Output Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### **Camera Calibration**

#### **1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.**

Related code - [camera.py].

- Generated 3D coordinates for object points (`coord3dObjPts`) using the corners shape *(9, 6)* and by assuming that the image is on the plane Z = 0.
- For each calibration chessboard image, image points are generated using `cv2.findChessboardCorners()` and these are appended to a list.
  - For some calibration images, image points could not be generated. This was mostly due to the calibration image not containing the required corners - (9, 6).
  - Such file names have been listed in the [log file].
- The object points coordinates (`coord3dObjPts`) remain same for each calibration image. Thus, these, too, are appended to a list every iteration.
- Once every calibration image has been processed, the list of image points and object points are passed to the `cv2.calibrateCamera()` function.
- This returns the camera intrinsic and extrinsic parameters (K, R and t) along with the distortion coefficients which are logged and saved as properties.
- These matrices are then used in the `undistort` member function which internally calls the `cv2.undistort()` function.
- Here's an example of a calibration image and the corresponding undistorted image -

![alt text][undistort1]

### **Pipeline (single image)**

#### **1. Provide an example of a distortion-corrected image.**

- Here's an example of an undistorted frame -
![alt text][undistort2]

#### **2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.**

</br>Related Code -

| Function | File   |
|:--------:|:------:|
| `detectEdges` | [lane.py] |
| `gradMagThreshold` | [helper.py] |
| `gradDirThreshold` |  [helper.py] |
| `gradThreshold` | [helper.py] |
| `colorThreshold` | [helper.py] |

- Color Thresholding
  - Tried to understand what the different channels in LAB and HLS did for different images and which would be the best choice for thresholding.
    ![alt text][hls_lab]
  - Chose Saturation channel as the other channels didn't offer anything better.
  - Tried adding yellow and white color masks but they too didnt add any additional benefits.
- Gradient Thresholding
  - Performed thresholding based on gradient magnitude (gradients both in X and Y direction) and direction.
- Final binary image is created by combining the output of Color and Gradient Thresholding.
- In the following example, S-Channel + Grad represents the final binary image.

![alt text][Thresholded]

#### **3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.**

</br>Related Code -

| Function | File   |
|:--------:|:------:|
| `perspectiveTransform` | [lane.py] |
| `warpImage` | [lane.py] |
| `unwarpImage` | [lane.py] |

- As a one time process, hardcoded source and destination points are passed to generate the warp and unwarp (warp inv) matrices.
- These points were chosen using the `straight_lines2.jpg` image to represent the car's straight field of view. Thus, such field of view would help recognize turns and curves easily.
- Here are the points -
| Source        | Destination   |
|:-------------:|:-------------:|
| 271, 680      | 300, 720        |
| 1062, 680      | 900, 720      |
| 690, 450     | 900, 0      |
| 597, 450      | 300, 0        |

- Perspective transform on a straight road -

![alt text][perspective_transform]

- Perspective transform on a curve -

![alt text][perspective_transform2]

#### **4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?**

Related code - [lane.py].

- There were mainly 2 approaches here - brute force using a histogram and sliding window (`__findLaneByHistogramSlidingWindow__`) and based on previous curve ROI (`__findLaneByPreviousCurve__`).
- Both of these methods would return the left and right lane information along with the decision confidence.
- If the previous curve information confidence was higher than the threshold, the algorithm would use that as a reference and thus, would search based on previous curve.
- However, if it was below the threshold then the histogram brute force logic would run.

- Brute-Force using Histogram and sliding window
  - Using the x between (0, mid) and that between (mid, width) with highest intensity sum (as indicated by a histogram over the bottom half) as the left and right lane starting points respectively, a sliding window of fixed height and width for left and right lanes, would start from the bottom of the image.
  - The non-zero pixels within these 2 rectangles (1 for left, 1 for right) would be marked as lane pixels.
  - A 2D curve would be fit over these lane pixels to get the left and right lane.
  - Here's an example -
   ![alt text][lane_roi1]

- Based on previous curve
  - Using the previous curve information as reference, the ROI is narrowed down to the curves and +- ROI margin.
  - Non-zero pixels within this ROI region would be considered lane pixels and would be used to fit a 2D curve.
  - However, the final curve information is a weighted sum of the new information and the previous curve information (line: 357, [lane.py]).
    - This helps to maintain an exponentially weighted sum of all the high-confidence curves generated.
  - Here's an example -
   ![alt text][lane_roi2]

#### **5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.**

- Related code - Function: `__calculateRadiusCurvature__` in [lane.py]

#### **6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.**

- Related code - Function: `__drawLane__` in [lane.py]
- This generates the lane masks and then unwarps them and draws them onto the original image along with the lane area marked in a different color.
- The output image also shows ROI image generated by the find lane methods, radius of curvature, warped edges and lanes as shown below -

   ![alt text][output]

---

### **Pipeline (video)**

#### **1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).**

- Here's a link to the output video - [output video]

<figure class="video_container" poster="./test_images/test3.jpg">
  <video controls="true" allowfullscreen="true">
    <source src="./output_images/out_project_video.mp4" type="video/mp4">
  </video>
</figure>

---

### **Discussion**

#### **1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?**

- Runtime - This algorithm is not realtime. Since most operations are iterative and don't involve any sort of parallelization, the overall speed of processing a frame is low. Current speed processes 25 frames (1 sec of video at 25 fps) in ~3 seconds.
- Occlusion - Currently, the initial binary image generation just depends on gradient and s-channel. So, any occlusion of the lanes or rather a vehicle in front, could come under the ROI area and be marked a lane.
- Shadows - While the dependence on S-channel helps handle minor lighting changes, drastic shadows and lighting differences do still appear on the thresholded view. Thus, in some cases, these shadows are considered lane pixels and end up messing the lane curve prediction.

Ideally, solely relying on this thresholding is not enough. While the Curve prediction algorithms are time consuming, they work as long as the input warped binary image is fine. Thus, using information from feature extractors, temporal factors or rather moving to some ML or DL based solution would make sense as they would be able to handle such outliers.
Also, there could be further improvements in the confidence logic, setting up lane rules (should be parallel +- error) etc.


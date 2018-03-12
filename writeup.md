# Vehicle Detection Project


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[//]: # (Image References)
[image1a]: ./writeup_pics/image0070.png
[image1]: ./examples/car_not_car.png
[image1b]: ./writeup_pics/image1022.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4
[image_car_ch]: ./writeup_pics/car_ch.png
[image_notcar_ch]: ./writeup_pics/notcar_ch.png
[image_car_hog]: ./writeup_pics/car_hog.png
[image_notcar_hog]: ./writeup_pics/notcar_hog.png

[heat1]: ./writeup_pics/heatnew1.png
[heat2]: ./writeup_pics/heatnew2.png
[heat3]: ./writeup_pics/heatnew3.png
[heat4]: ./writeup_pics/heatnew4.png
[heat5]: ./writeup_pics/heatnew5.png
[heat6]: ./writeup_pics/heatnew6.png

[labels1]: ./writeup_pics/labels1.png
[labels2]: ./writeup_pics/labels2.png
[labels3]: ./writeup_pics/labels3.png
[labels4]: ./writeup_pics/labels4.png

[last]: ./writeup_pics/last.png

[example1]: ./writeup_pics/example1.png
[example2]: ./writeup_pics/example2.png
[example3]: ./writeup_pics/example3.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is in file `hog_classify.py` lines #50 through the end of module

I started by reading in all the `vehicle` and `non-vehicle` images from folders (`read_image_folders()` function).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes.
Car:
![alt text][image1a]
Not car:
![alt text][image1b]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=2`:

Car image and it's hog:
![alt text][image_car_ch]
![alt text][image_car_hog]

Not car image and it's hog
![alt text][image_notcar_ch]
![alt text][image_notcar_hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and stepped on almost default ones. I read on forum, that better to pick `LUV` or `YCrCb` color channels. I tried a lot with HLS color scheme mentioned in lections but struggled to build useful model despite getting low loss score in SVC model after training. So I picked `LUV` color scheme after huge amount of exmpirements
My final parameters:

|       |     |
|----------|:-------------:|
|orient | 9   | HOG orientations
|pix_per_cell | 16  | HOG pixels per cell
|cell_per_block | 2  | HOG cells per block
|hog_channel | 2  | Can be 0, 1, 2, or "ALL"
|spatial_size | (32, 32)  | Spatial binning dimensions
|hist_bins | 32     | Number of histogram bins

These parameters can be found in line 64 to 70 in `hog_classify.py` module
I choose them by conducting huge amount of experiments. Pixels per cell can be 16 which gives us faster training and hog computation. I would describe parameters as a set which gives us result we need in reasonable time. We need to compute fast in order to keep experiments with parameters

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features. Alone HOG features didn't worked out for me, so after some experiments I added additional channels used in lections code (spatial channel and color histogram channel). After some debugging and testing model worked out!
I downloaded all pictures with vehicles and with non vehicles mentioned in lectures. Build up a 2 data sets, shuffle it and split on 2 part: training and testing. Right after I just fed data into `svc.fit` function which did all work for me :)
I saved result in pickle file to avoid training in every program run iteration.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Main module where code grabs video frame by frame called `pipeline.py`
I decided to use lection code with sliding window approach. I picked some ROI (less than half of image) and fullfill area with smaller squares characterized parameter `scale` which multiply window area (64, 64) to its factor. The more `scale` the bigger window. These windows fitted in ROI and possible overlap on 2 cell sizes, so the same area in picture will be covered several times by different windows, sometimes even with different sizes.
I also found good advice to change ystart and ystop for each scale differently. Thus we can avoid to compute small regions to close to our car (look at function `pipeline()` in module `pipeline.py` 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images with raw bounding boxes:

![alt text][example1]
![alt text][example2]
![alt text][example3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/ishipachev/UdacitySDCND-CarND-Vehicle-Detection-P5/blob/master/output/output.avi)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

To fight fasle positive I saved heatmap from previous frame for next 6 frames (line 52 module `heat.py`). Limit count for heat was picked as 9 (line 70 module `heat.py`). I saved frame and calculated heat for all boxes for current frame and for previous 5. So right after I can drop false positive blinking boxes and add some stability to bounding box for the car. Look for keep_heat function (line 50 to 61 module `heat.py`)

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][heat1]
![alt text][heat2]
![alt text][heat3]
![alt text][heat4]
![alt text][heat5]
![alt text][heat6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][labels1]
![alt text][labels2]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][last]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

1. I struggled alot with colorspaces and parameters.
2. Seems like simple neural network could solve same task to recognize cars on 32x32 picture but probably even faster.
3. All computations were performed on CPU and took enough amount of time. Probably there is the way to shift computations on GPU.
4. There is possibility to decrease computations just by removing small windows at the bottom of the picture and big ones at the top of the picture. We should expect big picture of a car far away neither small picture of car near of our car.
5. Stabilisation through frames worked pretty well and allowed me to keep non perfect model and use this trick to cut off almost all false negatives.
6. As you can see, sometimes vehicles on the counter lanes trigger our detector.

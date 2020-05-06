# Image processing pipeline for autonomous driving
## Overview
  This is an image processing pipeline based mainly on the OpenCV package. The core idea is to package all image processing techniques into class methods, so that the image processing would be much more straight-forward. Each feature to be processed corresponds to an instance - either an **ImageFeature**, an **ImageMask** or simply a **Canvas** depending on its type. Each such instance would have a canvas attribute, which is the image array to be drawn feature on.

  By calling the class methods of the instance, different operations would be performed on the canvas. For example, if we have an ImageFeature instance named if1 that has already been initialized with an image array. If we want to get the sobel derivative in x direction of the R channel, we can do this: `if1.channel_selection('R').gaussian_blur().sobel_convolute('x').binbinary_threshold((30,100), show_key=True)`

  For all features corresponding to the same picture, there is a cathegorizing class called **FeatureCollector**. An instance of this contains a python dict of all name: Canvas_instance pairs. It also supports different combination methods between different features to provide the final result we wish to have.
## Program Structure
  As mentioned above, the main body of the project are the **FeatureCollector** class and all kinds of classes derived from the **Canvas** superclass. For the time being, there are 6 kinds of Canvases:
  - Canvas2: *#The basic Canvas object for single channel features.*
    - attributes: **canvas**
    - added methods: **calibration**, **show_layer**
    - modified methods: __and__, __or__, __xor__, __add__


  - Canvas3(Canvas2): *#Extend the Canvas2 to 3-channel images.*
    - attributes: **canvas**
    - added methods: **Canvas2GRAY**
    - modified methods: __add__


  - ImgFeature2(Canvas2): *#Image features extracted from single channel images.*
    - attributes: **canvas**, **img**
    - added methods: **binary_threshold**, **gaussian_blur**, **sobel_convolute**


  - ImgFeature3(Canvas3, ImgFeature2): *#Image features extracted from 3-channel images.*
    - attributes: **canvas**, **img**
    - added methods: **channel_selection**


  - ImgMask2(Canvas2): *#Image features that aren't directly related to the image contents.*
    - attributes: **canvas**, **img**
    - added methods: **geometrical_mask**, **straight_lines**


  - ImgMask3(Canvas3, ImgMask2): *#Image features that aren't directly related to the image contents.*
    - attributes: **canvas**, **img**
    - added methods:


  - FeatureCollector: *#Collects and processes different features from the same image.*
    - attributes: **img**, **color_model**, **img_processed**, **layers_dict**
    - methods: **add_layer**, **get_chessboard_calibrators**, **image_show**
    - overloaded methods: __call__

## Typical Tasks
### 1. Lane Line Detection-Hough Transform
This project aims in detecting the simplest case of lanemark detection - finding well-painted straight lanelines on both sides of the ego-vehicle.

![image_sample](../test_images/solidWhiteCurve.jpg)

★★Before pipelining any images, the calibration parameters shall be calculated and stored inside the FeatureCollector.

The basic detecting scheme works in such a pipeline:

1. Use a **Canny lane detector** to detect all edges in the image.  
2. Extract geometrically the region of interest, in which both lane-lines appear.  
3. Use a **Hough transmitter** to transmitt all edges into single, long straight lines.  
4. Group all lines into 2 groups, representing 2 lanes on both sides.

A sample output is shown as below:

![output_result](../test_images_output/solidWhiteCurve.jpg)
### 2. Lane Line Detection-Sliding Window

import cv2
import os

from cam_image_process import lanes_detection_polyfit as ldsw

# Step1: Input the image and Preview
img1 = cv2.imread("./test_images/straight_lines1.jpg")

# Step2: Initialize the container FeatureCollector object.
name_list = []
cal_imgs_list = []
for path, dir_list, file_list in os.walk(r".\camera_cal"):
    for name in file_list:
        img_cal = cv2.imread(os.path.join(path, name))
        cal_imgs_list.append(img_cal)
fc = ldsw.container_initialization(img1, cal_imgs_list, show_key=False)

# Step3: Test on all 6 test images
for path, dir_list, file_list in os.walk(r".\test_images"):
    for name in file_list:
        if not name.startswith("test1"): continue
        img_test = cv2.imread(os.path.join(path, name))
        left_lane_params, right_lane_params, fc = ldsw.lane_detector_initial(fc, img_test, 2)
        y_range = [0, fc.img_processed.shape[0]]
        _,_,fc = ldsw.live_info_calculation(left_lane_params, right_lane_params, y_range, fc)
        left_lane_params, right_lane_params, fc = ldsw.lane_detector_sequential(fc, img_test, left_lane_params, right_lane_params)
        # fc.image_save("info_printed_" + name.split('.')[0])

# fc.image_show()
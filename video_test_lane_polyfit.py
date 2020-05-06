import cv2
import os
import numpy as np

from cam_image_process import lanes_detection_polyfit as ldsw

# Step1: Input the video
input_name = "challenge_video.mp4"
cap1 = cv2.VideoCapture("./test_videos/" + input_name)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter("./output_videos/processed_" + input_name, fourcc, 30.0, (1280, 720))

# Step2: Initialize the container FeatureCollector object.
name_list = []
cal_imgs_list = []
for path, dir_list, file_list in os.walk(r".\camera_cal"):
    for name in file_list:
        img_cal = cv2.imread(os.path.join(path, name))
        cal_imgs_list.append(img_cal)
fc = ldsw.container_initialization(img_cal, cal_imgs_list, show_key=False)

# Step3: Edit the video
## 3.1: Use the sliding window detector to detect the first frame
if(cap1.isOpened()):
    ret, frame = cap1.read()
    if ret:
        left_lane_params, right_lane_params, fc, inds_number = ldsw.lane_detector_initial(fc, frame, 2)
        lane_params = np.array([[left_lane_params, right_lane_params]])
        fc.lane_params_refresh(lane_params, inds_number)
        _,_,fc = ldsw.live_info_calculation(left_lane_params, right_lane_params, np.array([0, frame.shape[0]]), fc)
        out.write(fc.img_processed)
    else:
        print("Unable to Read Video Stream!")

## 3.2: Use the sequential detector to detect following frames
number_of_frames = 1
while(cap1.isOpened()):
    ret, frame = cap1.read()
    if ret:
        number_of_frames += 1
        left_lane_params, right_lane_params, fc, inds_number = ldsw.lane_detector_sequential(fc, frame, fc.lane_params_current[0], fc.lane_params_current[1])
        lane_params = np.array([[left_lane_params, right_lane_params]])
        fc.lane_params_refresh(lane_params, inds_number)
        _, _, fc = ldsw.live_info_calculation(fc.lane_params_current[0,:], fc.lane_params_current[1,:], np.array([0, frame.shape[0]]), fc)
        out.write(fc.img_processed)
        print(".", end='')
        if number_of_frames % 150 == 0:
            print(number_of_frames, "frames are processed")
            fc.image_show()
    else:
        break

# Release everything if job is finished
print("Convertion Finished!")
cap1.release()
out.release()
cv2.destroyAllWindows()

# fc.image_show()


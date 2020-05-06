"""
This file serves as settings for the lane detection project.
"""
import numpy as np

# Parameters related to I/O information
output_img_path = "./output_images/"
output_video_path = "../output_videos/"

# Parameters for the Canny Edge Detector
canny_mag_trs_low = 80
canny_mag_trs_high = 255
canny_mag_trs = (canny_mag_trs_low, canny_mag_trs_high)
canny_dir_trs_low = np.pi/5
canny_dir_trs_high = np.pi/3
canny_dir_trs = (canny_dir_trs_low, canny_dir_trs_high)
gaussian_apertureSize = 3
gaussian_kernel_size = (gaussian_apertureSize, gaussian_apertureSize)

# Parameters for the Hough Transmitter
hough_rho = 2  # distance resolution in pixels of the Hough grid
hough_theta = np.pi / 180  # angular resolution in radians of the Hough grid
hough_threshold = 40     # minimum number of votes (intersections in Hough grid cell)
hough_min_line_length = 75  # minimum number of pixels making up a line
hough_max_line_gap = 150    # maximum gap in pixels between connectable line segments

# Parameters for the sliding window transmitter
num_windows = 15
margin_pix = 100
min_pix_4_update = 20
thickness_of_line = 50

# Parameters relating to the traffic
vehicle_width = 1.9
lane_width = 3.7
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = lane_width/700 # meters per pixel in x dimension

# Parameters of the camera
x_pix = 1280
y_pix = 720
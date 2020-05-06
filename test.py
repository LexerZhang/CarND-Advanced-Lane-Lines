import cv2
import os

from cam_image_process import experiment as exp

img1 = cv2.imread("test_images/test5.jpg")

# exp.channel_selection(img1)

exp.text_channel(img1)
"""Provide some general image_functions for image processing."""
from datetime import datetime

import cv2
import numpy as np


def image_show(img, name='image'):
    """Show an image until any key is pushed."""
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_save(img, name="image", suffix=".jpg", path="./"):
    """
    Save a picture to the designated path with the name specified.
    Return the name of the file.
    """
    if name == "image":
        now = datetime.now()
        current_time = now.strftime("_%y%m%d-%H%M%S")
        name += current_time
    cv2.imwrite(path + name + suffix, img)
    return name+suffix


def weighted_img(initial_img, img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def draw_lines(img, lines, color=(0, 255, 0), thickness=6):
    """
    This function draws `lines` with `color` and `thickness`,
    returning a blank image with lines drawn onto.
    """
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def draw_multinomials(img, params, color=(255, 255, 255), thickness=3):
    """
    This function draws multinomial curves with color and thickness.
    :param img: the original image to be drawn onto
    :param color: color of the curve, tuple(3)
    :param thickness: thickness of the curve, float
    :return img: the image with curves drawn onto
    """
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0] // 10, dtype=np.int32)
    plotx = np.zeros_like(ploty, dtype=np.float64)
    for param in params:
        plotx *= ploty
        plotx += param
    plotx = np.array(plotx, dtype=np.int32)
    points = np.array((plotx, ploty)).T
    points = np.array((points,))
    img = draw_polylines(img, points, False, color, thickness)
    return img

def draw_polylines(img, points, closed=False, color=(255,255,255), thickness=3):
    """
    This function draws curves with color and thickness.
    :param img: the original image to be drawn onto
    :param points: the list of points, numpy.array(number_of_curves x number_of_points x 2
    :param closed: if the curves are to be closed
    :param color: the color of the curves
    :param thickness: the thickness
    :return: the image with curves drawn onto
    """
    img = cv2.polylines(img, points, closed, color, thickness)
    return img

def image_normalization(img, abs=True):
    """
    Scale an image to a uint8 array for representation.
    :param img: single or multi-channel image
    :return: the scaled image matrix in format uint8
    """

    if abs:
        img = np.abs(np.int16(img))
    val_max = img.max()
    val_min = img.min()
    if val_min==val_max:
        return img
    else:
        return np.uint8((img - val_min) * 255 / (val_max - val_min))


def get_vertices(img_BGR):
    """
    Accepts an RGB image array, return a tuple of (lines, vertices) for region selection.

    Input:
    img_RGB: 3-tunnel image array, with size of Height * Width * Tunnels

    Output:
    lines: cordinates list of all lines to be drawn, size: 1 * Number_of_Lines * 4
    vertices: cordinates numpy array of all vertices, size: 1 * Number_of_Vertices * 2
    """
    y_max, x_max, _ = img_BGR.shape
    # Assign cordinates for the 4 corners
    Point_Lower_Left = (round(0.01 * x_max), y_max - 1)
    Point_Lower_Right = (round(0.99 * x_max), y_max - 1)
    Point_Upper_Left = (round(0.415 * x_max), round(0.65 * y_max))
    Point_Upper_Right = (round(0.583 * x_max), round(0.65 * y_max))
    Point_list = [Point_Lower_Left, Point_Lower_Right,
                  Point_Upper_Right, Point_Upper_Left]
    line = []
    vertices = []
    for i in range(len(Point_list)):
        line.append(Point_list[0] + Point_list[1])
        vertices.append(Point_list[0])
        Point_list = Point_list[1:] + Point_list[:1]
    lines = [line]
    vertices = np.array(vertices)
    return vertices, lines

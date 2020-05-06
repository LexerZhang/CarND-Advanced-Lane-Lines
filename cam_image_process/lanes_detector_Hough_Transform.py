from . import parameters as prm
from .image_functions.image_process_pipeline import *


def Hough_image_lane_detector(img1):
    """
    A Pipeline for lane detection of a single image. Results would be saved.
    :param I_BGR: Original BGR image array object
    :return lane_line_list: BGR image array object with Hough Lines and Vertices drawn onto it.
    """
    img_container = FeatureCollector(img1)
    vertices, edge_lines = helper.get_vertices(img_container.img)

    img_container.add_layer("region_mask", 'mask')
    img_container.layers_dict["region_mask"].geometrical_mask(vertices)

    img_container.add_layer("edge_lines", "mask")
    img_container.layers_dict["edge_lines"].straight_lines(edge_lines)
    # Canny detection
    canny_edges = ImgFeature3(img_container.img)
    canny_edges.channel_selection('R').canny_detection(apertureSize=prm.gaussian_apertureSize)
    img_container.add_layer("canny_edges", layer=canny_edges)
    img_container.combine("canny_edges", "region_mask", "and")

    hough_lines_list = hough_lines(img_container.img_processed[:,:,0], prm.hough_rho, prm.hough_theta, prm.hough_threshold,
                                   prm.hough_min_line_length, prm.hough_max_line_gap)
    hough_lines_list = hough2lane_lines(hough_lines_list, img_container.img)
    img_container.add_layer("hough_lines", "mask")
    img_container.layers_dict["hough_lines"].straight_lines(hough_lines_list, color_BGR=(200, 200, 0))
    img_container.combine("main", "hough_lines", "mix")

    return img_container.img_processed



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns the coordinates of Hough lines.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    ##lane_line = hough2lane_lines(lines, img)
    ##line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    ##draw_lines(line_img, lane_line)
    return lines


def hough2lane_lines(hough_line_list, img):
    """
    According to given hough lines, calculate 2 lane lines which averages lane lines on both sides.
    """
    h_img, w_img, _ = img.shape
    kl_min = -10
    kl_max = -0.5
    kr_min = 0.5
    kr_max = 10

    # Group all lines into 2 lists. Each line should be listed together with its k.
    line_list_l = []
    line_list_r = []

    for hough_line in hough_line_list:
        k = (hough_line[0][3] - hough_line[0][1]) / (hough_line[0][2] - hough_line[0][0])
        if k > kl_min and k < kl_max:
            line_list_l.append(hough_line)
        elif k > kr_min and k < kr_max:
            line_list_r.append(hough_line)

    # Average all ks
    k_l = 0
    k_r = 0
    x_list_l = []
    y_list_l = []
    x_list_r = []
    y_list_r = []

    if len(line_list_l) > 0:
        for line in line_list_l:
            x_list_l.append(line[0][0])
            x_list_l.append(line[0][2])
            y_list_l.append(line[0][1])
            y_list_l.append(line[0][3])
            k_l = (max(y_list_l) - min(y_list_l)) / (max(x_list_l) - min(x_list_l))

    if len(line_list_r) > 0:
        for line in line_list_r:
            x_list_r.append(line[0][0])
            x_list_r.append(line[0][2])
            y_list_r.append(line[0][1])
            y_list_r.append(line[0][3])
            k_r = (max(y_list_r) - min(y_list_r)) / (max(x_list_r) - min(x_list_r))

    # Calculate the 2 x cordinates in the 2 lines to be returned
    line_l = [0, h_img - 1, 0, 0.6 * h_img]
    line_r = [w_img - 1, h_img - 1, w_img - 1, 0.6 * h_img]
    if k_l != 0:
        line_l[0] = min(x_list_l) - (line_l[1] - max(y_list_l)) / k_l
        line_l[2] = max(x_list_l) - (line_l[3] - min(y_list_l)) / k_l
    if k_r != 0:
        line_r[0] = min(x_list_r) + (line_r[1] - min(y_list_r)) / k_r
        line_r[2] = max(x_list_r) + (line_r[3] - max(y_list_r)) / k_r
    line_l = [map(int, line_l)]
    line_r = [map(int, line_r)]

    return np.array([line_l, line_r])


if __name__ == "__main__":
    img1 = cv2.imread('test_images/solidWhiteCurve.jpg')
    f_l = FeatureCollector(img1)
    f_l.image_show()

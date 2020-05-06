"""
This file contains a pipeline for lane detection with sliding windows.
"""

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from . import parameters as prm
from .image_functions.image_process_pipeline import *

__all__ = ["lane_detector_initial", "lane_detector_sequential", "container_initialization", "live_info_calculation"]


def container_initialization(img_raw, cal_imgs_list, warp=True, show_key=False):
    """
    Initialize a FeatureCollector with calibrators calculated from list_of_cal_imgs.
    Initialize the map_mat if necessary, too.
    :param img_raw: A sample image taken from the original camera.
    :param cal_imgs_list: A list of calibration image matrices.
    :param warp: Initialize the map_mat if True
    :return fc: The FeatureCollector object with calibrators preloaded.
    """
    ## 1.Camera Calibration and Image Undistortion
    vertices, lines = helper.get_vertices(img_raw)
    src_vertices = np.float32(vertices)
    fc = LaneFeatureCollector(img_raw)
    fc.get_chessboard_calibrators(cal_imgs_list, num_x=9, num_y=6, show_key=show_key)

    ## 2. Draw the edge lines for perspective transformation
    if warp: fc.get_warp_params(src_vertices)
    edge_lines = ImgMask3(img_raw)
    edge_lines.straight_lines(lines)
    fc.add_layer("edge_lines", "mask", False, edge_lines)
    fc.combine("main", "edge_lines", "mix")
    # fc.image_show() #: Comment
    # fc.image_save("edges_drawn") #: Comment

    if show_key:
        fc.undistort().warp("img_processed").image_show()
    return fc


def lane_detector_initial(fc, img_raw, degree_of_poly=2):
    """
    Takes a raw image as input and return the processed image, together with the curve parameters.
    :param img_raw: numpy.array of size hxwx3, the raw BGR image of the road.
    :param degree_of_poly: nint, the degree of polynomial curves of the lane lines
    :return img_processed: numpy.array of size hxwx3, the processed with lane lines drawn onto it.
    :return poly_params: numpy.array of size 2x(1+degree_of_poly), the polinomial parameters of both lane lines
    """

    # Step 1: Initialize the FeatureCollector with the warping parameters
    fc.image_reload(img_raw).undistort()

    # Step2: Edge Extracting pipeline
    fc = image_preprocess_pipeline(fc)

    # Step 3: Use a sliding window to detect lane lines
    ## Step 3.1: Get the polynomial parameters
    squares_list, left_lane_params, right_lane_params, number_of_inds = lane_finding_sliding_windows(
        fc.img_processed[:, :, 0],
        degree_of_poly)

    ## Step 3.2 Visualize the warped curve
    lane_curves_initial = ImgMask3(fc.img_processed)
    points_l = get_points(lane_curves_initial.canvas, left_lane_params)
    points_r = get_points(lane_curves_initial.canvas, right_lane_params)
    points_all = np.vstack((points_l, points_r[::-1, :]))
    points_list = np.array((points_all,))
    lane_curves_initial.geometrical_mask(points_list, (0, 50, 0))  # Fill in the lane region
    lane_curves_initial.polylines(np.array([points_l, ]), color=(0, 0, 255), thickness=prm.thickness_of_line).polylines(
        np.array([points_r, ]), color=(255, 0, 0), thickness=prm.thickness_of_line, show_key=True)  # Draw the lane lines
    # lane_curves_initial.save_layer("lane_curves_warped", path=prm.output_img_path) #: Comment

    ## Step 3.3 Unwarp the curves and add them to the initial image
    fc.add_layer("lane_curves_initial", layer=lane_curves_initial)
    fc.warp("lane_curves_initial", reverse=True)
    fc.combine("main", "lane_curves_initial", "mix", (1, 1, 0))
    fc.image_show() #: Comment
    # fc.image_save("lane_curves_drawn", path=prm.output_img_path) # Comment
    return left_lane_params, right_lane_params, fc ,number_of_inds


def lane_detector_sequential(fc, img_raw, left_lane_params=0, right_lane_params=0, degree_of_poly=2):
    """
    The sequential lane detector, must have non-zero lane parameters as input.
    :param fc: A feature collector instance, with calibrators initialized
    :param img_raw: numpy.array of size hxwx3, the raw BGR image of the road.
    :param left_lane_params: The polynomial parameters of the left lane.
    :param right_lane_params: The polynomial parameters of the right lane.
    :param degree_of_poly: nint, the degree of polynomial curves of the lane lines
    :return img_processed: numpy.array of size h x w x 3, the processed with lane lines drawn onto it.
    :return poly_params: numpy.array of size 2x(1+degree_of_poly), the polinomial parameters of both lane lines
    """
    # Step 1: Reload the FeatureCollector with a new image
    if left_lane_params is 0 and right_lane_params is 0:
        print("Detection is not initialized. Use the sliding windows method instead.")
        return lane_detector_initial(fc, img_raw, degree_of_poly)
    fc.image_reload(img_raw).undistort()

    # Step 2: Image Preprocess Pipeline
    fc = image_preprocess_pipeline(fc)

    # Step 3: Use previous curves to detect lane lines
    ## Step 3.1: Get the polynomial parameters
    lane_curves_sequential = ImgMask3(fc.img_processed)
    central_line_l = get_points(img_raw, left_lane_params)
    left_vertices, left_lane_inds = lane_segments_in_serpent(fc.img_processed[:, :, 0], central_line_l, prm.margin_pix)
    central_line_r = get_points(img_raw, right_lane_params)
    right_vertices, right_lane_inds = lane_segments_in_serpent(fc.img_processed[:, :, 0], central_line_r,
                                                               prm.margin_pix)
    serpent_vertices = np.array((left_vertices, right_vertices))

    left_lane_inds = reduce_to_line(left_lane_inds)
    right_lane_inds = reduce_to_line(right_lane_inds)
    leftx = left_lane_inds[1, :]  # .reshape(-1, 1)
    leftx_meter = leftx * prm.xm_per_pix
    lefty = left_lane_inds[0, :]  # .reshape(-1, 1)
    lefty_meter = lefty * prm.ym_per_pix
    rightx = right_lane_inds[1, :]  # .reshape(-1, 1)
    rightx_meter = rightx * prm.xm_per_pix
    righty = right_lane_inds[0, :]  # .reshape(-1, 1)
    righty_meter = righty * prm.ym_per_pix

    number_of_inds = np.array([len(leftx), len(rightx)])

    indices_mask = ImgMask3(fc.img_processed)
    indices_mask.fill_region(lefty, leftx, (0, 0, 255)).fill_region(righty, rightx, (
        255, 0, 0))  # .polylines(serpent_vertices, True, (255, 255, 0), show_key=False)  # Visulize the test result
    # indices_mask.save_layer("serpent_detection", path=prm.output_img_path)
    left_lane_params = np.polyfit(lefty_meter, leftx_meter, degree_of_poly)
    right_lane_params = np.polyfit(righty_meter, rightx_meter, degree_of_poly)

    ## Step 3.2 Visualize the warped curve
    points_l = get_points(lane_curves_sequential.canvas, left_lane_params)
    points_r = get_points(lane_curves_sequential.canvas, right_lane_params)
    points_all = np.vstack((points_l, points_r[::-1, :]))
    points_list = np.array((points_all,))
    lane_curves_sequential.geometrical_mask(points_list, (0, 50, 0))
    lane_curves_sequential.polylines(np.array([points_l, ]), color=(0, 0, 255),
                                     thickness=prm.thickness_of_line).polylines(
        np.array([points_r, ]), color=(255, 0, 0), thickness=prm.thickness_of_line)
    # lane_curves_sequential.save_layer("lane_curves_warped", path=prm.output_img_path)

    ## Step 3.3 Unwarp the curves and add them to the initial image
    fc.add_layer("lane_curves_sequential", layer=lane_curves_sequential)
    fc.warp("lane_curves_sequential", reverse=True)
    fc.combine("main", "lane_curves_sequential", "mix", (1, 1, 0))
    # fc.image_show()
    # fc.image_save("lane_curves_drawn", path=prm.output_img_path)
    return left_lane_params, right_lane_params, fc, number_of_inds


def image_preprocess_pipeline(fc):
    """
    Preprocess steps to extract edges.
    :param fc: A FeatureCollector Instance.
    :return: fc
    """
    vertices, _ = helper.get_vertices(fc.img)
    # fc.image_show()  # The original image

    ## 1. Create the trapezoid region mask
    region_mask = ImgMask3(fc.img)
    # vertices = np.array([vertices,])
    region_mask.geometrical_mask(vertices, ignore_mask_color=(255, 255, 255))  # , show_key=True)
    fc.add_layer("region_mask", "mask", layer=region_mask)

    ## 2. Create the binary image (a combinition of S and R channels)
    fc.add_layer("channel_R", "feature", use_calibrated=True)
    fc.add_layer("channel_S", "feature", use_calibrated=True)
    fc.layers_dict["channel_R"].channel_selection('R')
    fc.layers_dict["channel_S"].channel_selection('S')
    fc.combine("channel_R", "channel_S", "mix", (0.6, 0.9, -50))
    # channel_S_R = ImgFeature3(fc.img_processed)

    ## 3. The sobel magnitude binary
    sobel_mag = ImgFeature3(fc.img_processed)
    sobel_mag.gaussian_blur(k_size=prm.gaussian_kernel_size)  # , show_key=True)
    sobel_mag.sobel_convolute("mag").binary_threshold(prm.canny_mag_trs)  # , show_key=True)
    fc.add_layer("sobel_mag", layer=sobel_mag)

    ## 4. The sobel direction binary
    sobel_dir = ImgFeature3(fc.img_processed)
    sobel_dir.gaussian_blur(k_size=prm.gaussian_kernel_size)  # , show_key=True)
    sobel_dir.sobel_convolute("dir").binary_threshold(prm.canny_dir_trs)  # , show_key=True)
    fc.add_layer("sobel_dir", layer=sobel_dir)

    ## 5. Create the warped binary
    # fc.warp().image_show() # The warped original
    # fc.image_save("color_warped",path=prm.output_img_path)
    fc.combine("sobel_mag", "sobel_dir", "and")
    # fc.image_save("edges_extracted")
    fc.combine("sobel_mag", "region_mask", "and")
    # fc.image_show() # Comment

    fc.warp("img_processed")  # .image_show() # Comment
    # fc.image_save("binary_warped", path=prm.output_img_path) # Comment
    return fc


def lane_finding_sliding_windows(img_binary_warped, degree_of_poly=2):
    """
    Calculate both lane line curves' parameters.
    :param img_binary_warped: The warped binary image of road, numpy.array[h, w]
    :param degree_of_poly: degrees of curve to be calculated
    :return: left_fit, tuple(degree_of_curves + 1)
    :return: right_fit, tuple(degree_of_curves + 1)
    """
    ## 0. Initialize containers for lane pixels
    left_lane_inds = np.zeros((2, 0), dtype=np.int32)
    right_lane_inds = np.zeros_like(left_lane_inds, dtype=np.int32)
    squares_list = []

    ## 1. Use a histogram to calculate the first windows at bottom.
    histogram = np.sum(img_binary_warped[img_binary_warped.shape[0] // 3:, :], axis=0)
    middle_x = np.int(histogram.shape[0] / 2)
    leftx_center = np.argmax(histogram[:middle_x])
    rightx_center = np.argmax(histogram[middle_x:]) + middle_x

    ## 2. Iteratively add points to the lane pixels containers.
    win_height = img_binary_warped.shape[0] // prm.num_windows
    for i in range(prm.num_windows):
        win_y_low = img_binary_warped.shape[0] - i * win_height
        win_y_high = img_binary_warped.shape[0] - (i + 1) * win_height

        squares_list.append(get_window_edges(win_y_low, win_y_high, leftx_center, prm.margin_pix))
        squares_list.append(get_window_edges(win_y_low, win_y_high, rightx_center, prm.margin_pix))

        good_left_inds = lane_segments_in_slide(img_binary_warped, win_y_low, win_y_high,
                                                leftx_center, prm.margin_pix)
        good_right_inds = lane_segments_in_slide(img_binary_warped, win_y_low, win_y_high,
                                                 rightx_center, prm.margin_pix)
        left_lane_inds = np.concatenate((left_lane_inds, good_left_inds), axis=1)
        right_lane_inds = np.concatenate((right_lane_inds, good_right_inds), axis=1)
        if (good_left_inds.shape[1] > prm.min_pix_4_update):
            leftx_center = np.int(good_left_inds[1, :].mean())
        if (good_right_inds.shape[1] > prm.min_pix_4_update):
            rightx_center = np.int(good_right_inds[1, :].mean())

    ## 3. Reduce the lane lines into a single line.
    left_lane_inds = reduce_to_line(left_lane_inds)
    right_lane_inds = reduce_to_line(right_lane_inds)
    leftx = left_lane_inds[1, :]  # .reshape(-1, 1)
    leftx_meter = leftx * prm.xm_per_pix
    lefty = left_lane_inds[0, :]  # .reshape(-1, 1)
    lefty_meter = lefty * prm.ym_per_pix
    rightx = right_lane_inds[1, :]  # .reshape(-1, 1)
    rightx_meter = rightx * prm.xm_per_pix
    righty = right_lane_inds[0, :]  # .reshape(-1, 1)
    righty_meter = righty * prm.ym_per_pix

    number_of_inds = np.array([len(leftx), len(rightx)])

    ## An extra visualizing step

    indices_mask = ImgMask3(cv2.cvtColor(img_binary_warped, cv2.COLOR_GRAY2BGR))
    indices_mask.fill_region(lefty, leftx, (0, 0, 255)).fill_region(righty, rightx, (255, 0, 0)).straight_lines(
        squares_list, (255, 255, 0), 2)
    # indices_mask.save_layer("window_detection", path=prm.output_img_path)

    ## 4. Poly-fit the lane lines to get the polynomial parameters.
    left_fit = np.polyfit(lefty_meter, leftx_meter, degree_of_poly)  # TODO: Replace with sci-kit
    right_fit = np.polyfit(righty_meter, rightx_meter, degree_of_poly)  #
    return squares_list, left_fit, right_fit, number_of_inds


def live_info_calculation(left_fit, right_fit, y_range, fc):
    """
    Calculate the curvature and relative lateral position of the ego-vehicle and draw them onto the picture.
    :param left_fit: the polynomial coefficients of the left lane line
    :param right_fit: the polynomial coefficients of the right lane line
    :param y_range: numpy.array[2] the min/max value of y
    :param fc: a FeatureCollector object containing the image to be processed
    :return curvature: float, the curvature of the current lane line
    :return relative_position: float, the relative lateral position of the ego-vehicle in the lane line.
    :return fc: FeatureCollector object, containing the processed image.
    """
    ## 1. Calculate the curvature
    curvature_l = calculate_curvature(left_fit, y_range)
    curvature_r = calculate_curvature(right_fit, y_range)
    curvature = np.mean((curvature_l, curvature_r))
    curvature_text = "Radius of Curvature = " + str(round(curvature, 2)) + "(m)"

    ## 2. Calculate the relative lateral position
    l_position = calculate_relative_position(left_fit, y_range)
    position_text = "Vehicle is " + str(round(l_position, 2)) + "(m) from the left lane line"

    ## 3. Draw the information onto the image
    text_mask = ImgMask3(fc.img)
    text_mask.puttext(curvature_text, (30, 60))
    text_mask.puttext(position_text, (30, 120))
    text_mask.puttext("By Lix, ZHANG", (30, 180))
    fc.add_layer("text", layer=text_mask)
    fc.combine("main", "text", "or")
    # fc.image_show()

    return curvature, l_position, fc


def calculate_curvature(poly_coefs, y_range):
    """
    Calculate the maximum curvature within a curve segment.
    :param poly_coefs: the polynomial coefficients of the curvature
    :param y_range: numpy.array[2] the min/max value of y
    :return curvature: float, the maximum curvature
    """
    r = (1 + (2 * poly_coefs[0] * np.array(y_range) + poly_coefs[1]) ** 2) ** (3 / 2) / np.abs(2 * poly_coefs[0])
    return r.max()


def calculate_relative_position(poly_coefs_l, y_range):
    """
    Calculate the relative lateral position of the ego-vehicle in a lane.
    :param poly_coefs_l: the polynomial coefficients of the left lane
    :param poly_coefs_r: the polynomial coefficients of the right lane
    :param y_range: numpy.array[2] the min/max value of y
    :return: float, the relative position in the lane
    """
    vehicle_middle_x_bottom = prm.x_pix / 2
    left_lane_x_bottom = (poly_coefs_l[0] * (prm.ym_per_pix * y_range[1]) ** 2 + poly_coefs_l[1] * (
            prm.ym_per_pix * y_range[1]) + poly_coefs_l[2]) / prm.xm_per_pix

    return (vehicle_middle_x_bottom - left_lane_x_bottom) * prm.xm_per_pix - prm.vehicle_width / 2


def get_window_edges(win_y_low, win_y_high, x_center, margin):
    edges_list = []
    edges_list.append((x_center - margin, win_y_low, x_center + margin, win_y_low))
    edges_list.append((x_center + margin, win_y_low, x_center + margin, win_y_high))
    edges_list.append((x_center + margin, win_y_high, x_center - margin, win_y_high))
    edges_list.append((x_center - margin, win_y_high, x_center - margin, win_y_low))
    return edges_list


def get_points(img, params):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0] // 10, dtype=np.int32)
    plotx = np.zeros_like(ploty, dtype=np.float64)
    for param in params:
        plotx *= (prm.ym_per_pix * ploty)
        plotx += param
    plotx = np.array(plotx / prm.xm_per_pix, dtype=np.int32)
    points = np.array((plotx, ploty)).T
    return points


def lane_segments_in_slide(img_binary_warped, win_y_low, win_y_high, x_center, margin):
    win_x_left = x_center - margin
    win_x_right = x_center + margin
    mask = np.zeros_like(img_binary_warped)
    mask[win_y_high:win_y_low, win_x_left:win_x_right] = 1
    window = cv2.bitwise_and(img_binary_warped, mask)
    good_points = np.array(window.nonzero())
    return good_points


def reduce_to_line(inds_original):
    """
    Reduce the original indices to a single line such that the y coordinates are unique.
    :param inds_original: int np.array[2][], where inds_original[0,:] refers to the y-axis in image.
    :return: inds_reduced: int np.array[2][]
    """
    y_set = set(inds_original[0, :])
    inds_reduced = np.array((list(y_set), list(y_set)))
    for i in range(len(y_set)):
        same_y_inds = inds_original[0, :] == inds_reduced[0, i]
        inds_reduced[1, i] = np.int((inds_original[1, same_y_inds].max() + inds_original[1, same_y_inds].min()) / 2)
    return inds_reduced


def ridge_regression(x, y, degree, regularization_factor=0.1):
    """
    Use the sklearn machine learning package to do a ridge regression on the x and y labels.
    :param x: The x coordinates
    :param y: The y coordinates
    :param degree: degree of polynomial to be regressed
    :return: np.array[], a list of polynomial parameters
    """
    poly_model = make_pipeline(PolynomialFeatures(degree), Ridge(regularization_factor))
    poly_model.fit(y, x)
    params = poly_model
    print(params)
    return params


def lane_segments_in_serpent(img_binary_warped, central_line, margin):
    mask_layer = ImgMask2(img_binary_warped)
    margin_array = np.array([margin, 0])
    left_line = central_line - margin_array
    right_line = central_line + margin_array
    vertices = np.vstack((left_line, right_line[::-1, :]))
    vertices = np.array((vertices,))
    mask_layer.geometrical_mask(vertices)
    window = cv2.bitwise_and(img_binary_warped, mask_layer.canvas)
    good_points = np.array(window.nonzero())
    return vertices[0], good_points

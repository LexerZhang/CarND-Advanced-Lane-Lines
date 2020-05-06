"""Provide basic models to build up image pipelines."""
import cv2
import numpy as np

from . import helper


class Canvas2:
    """
    The superclass for any single-channel image elements.
    :self.attribute canvas: a numpy image matrix of the same size as the image to be processed.
    """

    def __init__(self, img):
        """
        set the self.img as a black image of the same size as img.
        :param img: the feature image numpy matrix
        """
        self.img = img
        self.canvas = np.zeros_like(img)

    def show_layer(self, key=False):
        if key:
            helper.image_show(self.canvas)

    def save_layer(self, name="image", suffix = ".jpg",path="./"):
        img_name = helper.image_save(self.canvas, name,suffix, path)
        print("image saved as", img_name)
        return self

    def img_normalization(self):
        self.canvas = helper.image_normalization(self.canvas)

    def __and__(self, other):
        """Return the bitwise-and result of 2 image matrices"""
        self.canvas = helper.image_normalization(cv2.bitwise_and(self.canvas, other.canvas))
        return self.canvas

    def __or__(self, other):
        """Return the bitwise-or result of 2 image matrices"""
        self.canvas = helper.image_normalization(cv2.bitwise_or(self.canvas, other.canvas))
        return self.canvas

    def __xor__(self, other):
        """Return the bitwise-xor result of 2 images matrices"""
        self.canvas = helper.image_normalization(cv2.bitwise_xor(self.canvas, other.canvas))
        return self.canvas

    def __add__(self, other):
        """Combine the 2 image features by setting them to 2 color channels."""
        self.canvas = helper.image_normalization(
            np.stack((self.canvas, other.canvas, np.zeros_like(self.canvas)), axis=2))
        return self.canvas


class Canvas3(Canvas2):
    """
    The superclass for any three-channel image elements.
    :self.attribute canvas: an image matrix of the same size as the image to be processed.
    :self.attribute canvas3: the original 3-channel BGR image matrix.
    """

    def Canvas2GRAY(self):
        """Turn the Canvas to gray scale image."""
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY).cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
        return self

    def __add__(self, other):
        """Combine the 2 image features by setting them to 2 color channels."""
        self.canvas = helper.image_normalization(
            np.stack((self.canvas[0], other.canvas[0], np.zeros_like(self.canvas[0])), axis=2))
        return self.canvas


class ImgFeature2(Canvas2):
    """
    The class takes in a single-channel image matrix and can do multiple openCV operations on the image.
    :self.attribute canvas: the feature image(single channel)
    :self.attribute img: the original image(single channel)
    """

    def __init__(self, img):
        """
        The initialization takes in an image matrix.
        For 2-dim images, format must be GRAY;
        for 3-dim images, format must be BGR.

        :param img: 2-dim image matrix
        """
        Canvas2.__init__(self, img)
        self.img = img
        self.canvas = img.copy()

    def binary_threshold(self, thresholds=(0, 255), show_key=False):
        """Create a binary image, in which 0 refers to the region within the thresholds. """
        self.canvas = helper.image_normalization((self.canvas > thresholds[0]) & (self.canvas < thresholds[1]))
        self.show_layer(show_key)
        return self

    def gaussian_blur(self, sigma=10, k_size=(3, 3), show_key=False):
        """Use a Gaussian Kernel to blur the image"""
        self.canvas = cv2.GaussianBlur(self.canvas, k_size, sigma)
        self.show_layer(show_key)
        return self

    def sobel_convolute(self, method, k_size=3, show_key=False):
        """
        Use a Sobel kernel to calculate the derivative of the image.
        """
        img_gray = self.canvas
        if method == 'x':
            self.canvas = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size))
        elif method == 'y':
            self.canvas = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size))
        elif method == 'dir':
            dx_img_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size))
            dy_img_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size))
            self.canvas = np.arctan2(np.absolute(dy_img_sobel), np.absolute(dx_img_sobel))
        else:
            dx_img_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size))
            dy_img_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size))
            self.canvas = np.sqrt(np.square(dx_img_sobel) + np.square(dy_img_sobel))
        # self.img_normalization()
        self.show_layer(show_key)
        return self

    def canny_detection(self, threshold_low=0, threshold_high=255, apertureSize=3, show_key=False):
        """
        Apply a canny detector the the image.
        :param threshold_low: integer 0~255, the lower threshold for canny detection
        :param threshold_high: integer 0~255, the higher threshold for canny detection
        """
        self.canvas = cv2.Canny(self.canvas[:, :, 0], threshold_low, threshold_high, apertureSize=apertureSize)
        self.show_layer(show_key)
        return self


class ImgFeature3(Canvas3, ImgFeature2):
    """
    The class takes in a three-channel image matrix and append some channel generating image_functions
    to the ImageFeature2 superclass.
    :self.attribute img: the feature image(single channel)
    """

    def __init__(self, img):
        ImgFeature2.__init__(self, img)

    def binary_threshold(self, thresholds=((0, 0, 0), (255, 255, 255)), show_key=False):
        """For 3-channel images, thresholds can be tuples."""
        return ImgFeature2.binary_threshold(self, thresholds, show_key)

    def channel_selection(self, label, show_key=False):
        """
        Get the specified channel image.
        :param label: Supported labels:
                      ('R', 'G', 'B', 'H', 'L', 'S')
        """
        if label.upper() == 'GRAY':
            self.canvas = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        elif label.upper() in 'BGR':
            self.canvas = self.img[:, :, 'BGR'.index(label)]
        elif label.upper() in 'HLS':
            self.canvas = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:, :, 'HLS'.index(label)]
        else:
            print("Sorry but this channel is not supported, return GRAY Scale instead.")
            self.canvas = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
        self.img_normalization()
        self.show_layer(show_key)
        return self

    def canny_detection(self, threshold_low=0, threshold_high=255, apertureSize=3, show_key=False):
        """
        Apply a canny detector the the image.
        :param threshold_low: integer 0~255, the lower threshold for canny detection
        :param threshold_high: integer 0~255, the higher threshold for canny detection
        """
        ImgFeature2.canny_detection(self, threshold_low, threshold_high, apertureSize=apertureSize)
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
        self.show_layer(show_key)
        return self


class ImgMask2(Canvas2):
    """
    Create an binary image mask using different kinds of edge extraction techniques.
    :self.attribute img: an image mask matrix of the same size as the image to be processed.
    """

    def geometrical_mask(self, vertices, show_key=False):
        """
        mask out the region outside of the vertices.
        :param vertices: numpy matrix of vertices, size: num_vertices x num_edges x 2
        """
        mask = np.zeros_like(self.canvas)  # defining a blank mask to start with
        ignore_mask_color = 255
        vertices = np.array([vertices,], np.int32)
        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        self.canvas = mask
        if show_key:
            self.show_layer(show_key)
        return self

    def fill_region(self, fill_x, fill_y, color=255, show_key=True):
        """
        Fill in a certain region with the color designated.
        :param fill_x: the x coordinates of pixels to be filled in
        :param fill_y: the y coordinates of pixels to be filled in
        :param color: the color of the fill-in
        """
        self.canvas[fill_x, fill_y] = color
        if show_key:
            self.show_layer(show_key)
        return self

    def straight_lines(self, lines, color=(255, 255, 255), thickness=6, show_key=False):
        """Create a mask with lines drawn with the parameters provided."""
        self.canvas = helper.draw_lines(self.canvas, lines, color, thickness)
        if show_key:
            self.show_layer(show_key)
        return self

    def curves(self, params, color=(255, 255, 255), thickness=3, show_key=False):
        """
        Create a mask with curves draen with the parameters provided
        :param params: the parameters of the curve, tuple(degree_of_curve+1)
        :param color: color of the curve, tuple(3)
        :param thickness: thickness of the curve, float
        :param show_key: bool
        """
        self.canvas = helper.draw_multinomials(self.canvas, params, color, thickness)
        if show_key:
            self.show_layer(show_key)
        return self


    def polylines(self, vertices, closed=False, color=(255,255,255), thickness=3, show_key=False):
        self.canvas = helper.draw_polylines(self.canvas, vertices, closed, color, thickness)
        if show_key:
            self.show_layer(show_key)
        return self


    def puttext(self, text, position=(30, 60), size=1.5, color=(255,255,255), thickness=4, show_key=False):
        """
        Put some text onto the image.
        :param text: string, the text string
        :param position: tuple[2], gives the position of the left-bottom corner
        :param size: uint, the size of the text
        :param color: tuple[3], BGR value of the color the text
        :param thickness: thickness of text lines
        """
        cv2.putText(self.canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
        if show_key:
            self.show_layer(show_key)
        return self


class ImgMask3(Canvas3, ImgMask2):
    """Image mask in 3 channels"""

    def __init__(self, img):
        ImgMask2.__init__(self, img)

    def geometrical_mask(self, vertices, ignore_mask_color=(0,255,0), show_key=False):
        """
        mask out the region outside of the vertices.
        :param vertices: numpy matrix of vertices, size: num_vertices x num_edges x 2
        """
        vertices = np.array([vertices,], np.int32)
        if len(self.img.shape) == 2:
            ignore_mask_color = 255
        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(self.canvas, vertices, ignore_mask_color)
        self.show_layer(show_key)
        return self

    def fill_region(self, fill_x, fill_y, color=(0, 0, 255), show_key=False):
        """
        Fill in a certain region with the color designated.
        :param fill_x: the x coordinates of pixels to be filled in
        :param fill_y: the y coordinates of pixels to be filled in
        :param color: the color of the fill-in
        """
        ImgMask2.fill_region(self, fill_x, fill_y, color, show_key)
        return self

    def straight_lines(self, lines, color_BGR=(0, 0, 255), thickness=6, show_key=False):
        """Create a mask with lines drawn with the parameters provided."""
        return ImgMask2.straight_lines(self, lines, color_BGR, thickness, show_key)

    def curves(self, params, color_BGR=(0, 255, 0), thickness=3, show_key=False):
        """
        Create a mask with curves draen with the parameters provided
        :param params: the parameters of the curve, tuple(degree_of_curve+1)
        :param color: color of the curve, tuple(3)
        :param thickness: thickness of the curve, float
        :param show_key: bool
        """
        return ImgMask2.curves(self, params, color_BGR, thickness, show_key)


class FeatureCollector:
    """
    Collects a list of features extracted from a single image.
    Use them for showing, combination, or simply acts as a pipeline.
    :self.attribute img: the BGR or GRAY image matrix
    :self.attribute layers_dict: list of image_feature instance
    :self.attribute color_model: the color model of image
    :self.attribute cameraMtx: camera matrix for calibration
    :self.attribute dist_coef: distortion coefficients for calibration
    """

    def __init__(self, img, color_model='BGR', calibrators=(0, 0, 0, np.eye(3))):
        """
        The initialization takes in an image matrix.
        Acceptable formats including:
            GRAY scale
            all validate color formats supported by openCV
        Images would be in default stored as uint8 format in BGR or GRAY.
        If the format is not BGR for a 3-dim image, a [format] must be assigned.

        :param img: 2-dim or 3-dim image matrix
        :param color_model: labels among: BAYER_BG, HLS, HSV, LAB, RGB, BGR, GRAY...
        :param calibrators: calibration parameters list following the order(number of
            chessboard images fed, Camera Matrix, Distortion Coefficient, Warp Matrix)
        """

        self.img = helper.image_normalization(img)
        self.img_processed = self.img.copy()
        self.layers_dict = {}
        # self.add_layer('main_canvas', "mask")
        self.calibrators = {"number_of_img": calibrators[0], "CamMtx": calibrators[1],
                            "DistCoe": calibrators[2], "WarpMtx": calibrators[3]}
        if len(self.img.shape) == 2:
            self.color_model = 'GRAY'
        elif color_model != 'BGR':
            l_valid_color_format = [key for key in cv2.__dict__.keys()
                                    if key.startswith('COLOR')
                                    and key.endswith('2BGR')
                                    and len(key.split('_')) == 2]
            if color_model in l_valid_color_format:
                cvt_method = "cv2.COLOR_" + color_model + "2BGR"
                self.img = cv2.cvtColor(self.img, eval(cvt_method))
            else:
                print('Unknown color model, please manually transfer to BGR.')
        self.color_model = 'BGR'
        self.add_layer("main", "feature")

    def image_reload(self, img):
        """
        Reload an image into the instance. Calibrators, color_model would be keeped.
        :param img:
        """
        self.img = helper.image_normalization(img)
        self.img_processed = self.img.copy()
        self.layers_dict = {}
        self.add_layer("main", "feature")
        return self

    def add_layer(self, key='layer', type='feature', use_calibrated=False, layer=None):
        """Add a new key:ImgFeature/ImgMask instance to the self.layers_dict."""
        if key == 'layer':
            key = 'layer_' + str(len(self.layers_dict))
        if layer is not None:
            self.layers_dict[key] = layer
        else:
            if use_calibrated and self.layers_dict["calibrated"] is not None:
                img_template = self.layers_dict["calibrated"].img
            else:
                img_template = self.img
            if type == 'feature':
                if self.color_model == "GRAY":
                    self.layers_dict[key] = ImgFeature2(img_template)
                else:
                    self.layers_dict[key] = ImgFeature3(img_template)
            else:
                if self.color_model == "GRAY":
                    self.layers_dict[key] = ImgMask2(img_template)
                else:
                    self.layers_dict[key] = ImgMask3(img_template)

    def get_chessboard_calibrators(self, chessboard_imgs_list, num_x=2, num_y=2, show_key=False):
        """
        Get calibrators using a chessboard image and the specified number of corners.
        By inputting an image, which is laid on the surface on which the features are,
        and shot by the original camera lens, this function would calculate out the parameters
        for both image undistortion and perspective transformation.
        :param chessboard_imgs_list: A list of chess board images
        :param corners_number: The number of corners on chessboard in x and y directions
        """
        obj_points = []
        img_points = []
        number_of_imgs = 0

        objp = np.zeros((1, num_x * num_y, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)

        for chessboard_img in chessboard_imgs_list:
            chessboard_img_gray = cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(chessboard_img_gray, (num_x, num_y), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                number_of_imgs += 1
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(chessboard_img_gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                img_points.append(corners2)
        else:
            print("Unable to detect corners for this image!")

        ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points,
                                                       chessboard_img_gray.shape[::-1], None, None)
        self.calibrators["number_of_img"] = number_of_imgs
        self.calibrators["CamMtx"] = mtx
        self.calibrators["DistCoe"] = dist

        i = 0
        for chessboard_img in chessboard_imgs_list[::5]:
            i+=1
            chessboard_img_gray = cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2GRAY)
            chessboard_undistorted_gray = cv2.undistort(chessboard_img_gray, mtx, dist)
            # helper.image_save(chessboard_img, "chessboard"+str(i))
            # helper.image_save(chessboard_undistorted_gray, "chessboard_undistorted"+str(i))
            if show_key:
                helper.image_show(chessboard_undistorted_gray)


    def get_warp_params(self, src_vertices):
        """
        Get the warp matrix with a source vertice.
        :param src_vertices:
        :return:
        """
        y_m, x_m, _ = self.img.shape
        margin = 10
        dst = np.array([[margin, y_m - margin], [x_m - margin, y_m - margin], [x_m - margin, margin], [margin, margin]],
                       dtype=np.float32)
        warp_mtx = cv2.getPerspectiveTransform(src_vertices, dst)
        self.calibrators["WarpMtx"] = warp_mtx

    def undistort(self, key="img"):
        """
        Undistort the image using the provided parameters.
        If key == "img", the undistorted will be stored under the key "calibrated",
        else, only the img_processed will be undistorted.
        """
        if self.calibrators["number_of_img"] == 0:
            print("Please calibrate with a chessboard image first. Undistortion will not be conducted. ")
            return self
        if key.lower()=="img":
            img_undistorted = cv2.undistort(self.img, self.calibrators["CamMtx"], self.calibrators["DistCoe"])
            try:
                self.layers_dict["calibrated"].img = img_undistorted
            except:
                self.layers_dict["calibrated"] = ImgFeature3(img_undistorted)
        else:
            img_undistorted = cv2.undistort(self.img_processed, self.calibrators["CamMtx"], self.calibrators["DistCoe"])
        self.img_processed = img_undistorted
        return self


    def warp(self, key="img", reverse=False):
        """
        Warp the image using a perspective transformation matrix.
        If key == "img", the unwarped will be stored under the key "calibrated",
        else, unwarp the designated key.
        :param key: the content to be warped
        """
        WarpMtx = self.calibrators["WarpMtx"]
        if reverse:
            WarpMtx = np.linalg.inv(WarpMtx)
        if key.lower()=="img":
            try:
                img_warped = cv2.warpPerspective(self.layers_dict["calibrated"].img, WarpMtx, self.img.shape[1::-1])
                self.layers_dict["calibrated"].img = img_warped
            except:
                img_warped = cv2.warpPerspective(self.img, WarpMtx, self.img.shape[1::-1])
                self.layers_dict["calibrated"] = ImgFeature3(img_warped)
        elif key.lower()=="img_processed":
            img_warped = cv2.warpPerspective(self.img_processed, WarpMtx, self.img.shape[1::-1])
        else:
            try:
                img_warped = cv2.warpPerspective(self.layers_dict[key].canvas, WarpMtx, self.img.shape[1::-1])
                self.layers_dict[key].canvas = img_warped
            except:
                print("Invalid Keys in warpping! Return warpping of original image instead")
                img_warped = cv2.warpPerspective(self.img, WarpMtx, self.img.shape[1::-1])
        self.img_processed = img_warped
        # self.image_save("image_warped")
        return self

    def image_show(self, show_key=True):
        if show_key:
            helper.image_show(self.img_processed)

    def image_save(self, name="image", suffix=".jpg", path="./"):
        img_name = helper.image_save(self.img_processed, name, suffix, path)
        print("image saved as", img_name)

    def combine(self, key1, key2, method='and', parameter=(1, 0.5, 0)):
        """
        Return the Combination of 2 features in the self.layers_dict according to the method.
        :param key1, key2: The keys of canvases to be combined.
        :param method: Choose from("and", "or", "xor", "add")
        """
        try:
            layer1 = self.layers_dict[key1]
            layer2 = self.layers_dict[key2]
        except:
            print("Invalid keys!")
            return
        if method == 'and':
            self.img_processed = layer1 & layer2
        elif method == 'or':
            self.img_processed = layer1 | layer2
        elif method == 'xor':
            self.img_processed = layer1 ^ layer2
        elif method == 'add':
            self.img_processed = layer1 + layer2
        elif method == 'mix':
            self.img_processed = helper.weighted_img(layer1.canvas, layer2.canvas, parameter[0], parameter[1],
                                                     parameter[2])
            self.layers_dict[key1].canvas = self.img_processed
        else:
            print("Doesn't support such method, sorry.")
            return

class LaneFeatureCollector(FeatureCollector):
    """
    This class adds some attributes for storing lane coefficients.
    """
    def __init__(self, img, color_model='BGR', calibrators=(0, 0, 0, np.eye(3)), params_cache_size=10):
        FeatureCollector.__init__(self, img, color_model, calibrators)
        self.lane_params_list = np.array([])
        self.lane_params_current = np.array([])
        self.inds_number_list = np.array([])
        self.params_cache_size = params_cache_size

    def lane_params_refresh(self, lane_params, inds_number):
        """
        Organizes the lane_params_list
        :param lane_params: np.array[][2][], the incoming polynomial coefficients
        :param inds_number: np.array[2], the incoming polynomial coefficients
        """
        # Step1: Refresh the lane parameters list
        inds_number = np.array([[[inds_number[0]], [inds_number[1]]]])
        if len(self.lane_params_list) <=0 :
            self.lane_params_list = lane_params
            self.inds_number_list = inds_number
        else:
            self.lane_params_list = np.concatenate((self.lane_params_list, lane_params))
            self.inds_number_list = np.concatenate((self.inds_number_list, inds_number))
        if len(self.lane_params_list) >= self.params_cache_size:
            self.lane_params_list = self.lane_params_list[1:]
            self.inds_number_list = self.inds_number_list[1:]

        # Step2: Calculate the current lane parameters
        lane_params_weighted = self.lane_params_list * self.inds_number_list
        self.lane_params_current = lane_params_weighted.sum(axis=0) / self.inds_number_list.sum(axis=0)
        self.lane_params_list[-1] = self.lane_params_current
        self.inds_number_list[-1] = self.inds_number_list.mean(axis=0)

        return self

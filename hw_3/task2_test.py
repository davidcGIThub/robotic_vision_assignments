import cv2 as cv
from matplotlib.pyplot import gray
import numpy as np
import glob
import os


# task 2 test
import cv2 as cv
import numpy as np
import glob
import os

#task 2
# finds the calibration parameters for the camera test images

#Load parameters

left_camera_matrix = np.load("left_intrinsic_parameters_test.npy")
left_distortion_parameters = np.load("left_distortion_parameters_test.npy")
right_camera_matrix = np.load("right_intrinsic_parameters_test.npy")
right_distortion_parameters = np.load("right_distortion_parameters_test.npy")

# Left Camera
image_directory = os.getcwd() + "/Stereo_Calibration_Practice_Images/Practice/SL"
data_path = os.path.join(image_directory,'*.bmp')
files = sorted(glob.glob(data_path))
number_of_inner_column_corners = 10
number_of_inner_column_rows = 7
chessboard_shape = (number_of_inner_column_corners,number_of_inner_column_rows)
zero_zone = (-1,-1)
half_of_window_size = (11,11)
termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
# termination_criteria = None
world_points = np.zeros((number_of_inner_column_corners*number_of_inner_column_rows,3), np.float32)
world_points[:,:2] = np.mgrid[0:number_of_inner_column_corners,0:number_of_inner_column_rows].T.reshape(-1,2)*2
world_points_array = []
left_image_points_array = []
for file in files:
    gray_image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    return_value, corner_locations = cv.findChessboardCorners(gray_image, chessboard_shape)
    refined_corner_locations = cv.cornerSubPix(gray_image, corner_locations, half_of_window_size, zero_zone,termination_criteria)
    world_points_array.append(world_points)
    left_image_points_array.append(refined_corner_locations)

# 
image_directory = os.getcwd() + "/Stereo_Calibration_Practice_Images/Practice/SR"
data_path = os.path.join(image_directory,'*.bmp')
files = sorted(glob.glob(data_path))
number_of_inner_column_corners = 10
number_of_inner_column_rows = 7
chessboard_shape = (number_of_inner_column_corners,number_of_inner_column_rows)
zero_zone = (-1,-1)
half_of_window_size = (11,11)
termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
# termination_criteria = None
right_image_points_array = []
for file in files:
    gray_image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    return_value, corner_locations = cv.findChessboardCorners(gray_image, chessboard_shape)
    refined_corner_locations = cv.cornerSubPix(gray_image, corner_locations, half_of_window_size, zero_zone,termination_criteria)
    right_image_points_array.append(refined_corner_locations)
image = cv.imread(image_directory + "/StereoR00.bmp", cv.IMREAD_GRAYSCALE)

flags = cv.CALIB_FIX_INTRINSIC
# finds the calibration parameters for the camera test images
ret_val, l_cam_matrix, l_dist_param, r_cam_matrix, r_dist_param, rotation, translation , essential , fundamental = cv.stereoCalibrate(world_points_array, left_image_points_array, 
                                                    right_image_points_array, left_camera_matrix, left_distortion_parameters,
                                                    right_camera_matrix, right_distortion_parameters,np.shape(image),flags)
print("rotation: " , rotation)
print("translation: " , translation)
print("essential matrix: " , essential)
print("fundamental: " , fundamental)

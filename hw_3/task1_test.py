import cv2 as cv
import numpy as np
import glob
import os

#task 2
# finds the calibration parameters for the camera test images

# Left Camera
image_directory = os.getcwd() + "/Stereo_Calibration_Practice_Images/Practice/L"
data_path = os.path.join(image_directory,'*.bmp')
files = glob.glob(data_path)
number_of_inner_column_corners = 10
number_of_inner_column_rows = 7
chessboard_shape = (number_of_inner_column_corners,number_of_inner_column_rows)
zero_zone = (-1,-1)
half_of_window_size = (11,11)
termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
world_points = np.zeros((number_of_inner_column_corners*number_of_inner_column_rows,3), np.float32)
world_points[:,:2] = np.mgrid[0:number_of_inner_column_corners,0:number_of_inner_column_rows].T.reshape(-1,2)
world_points_array = []
image_points_array_left = []
for file in files:
    gray_image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    return_value, corner_locations = cv.findChessboardCorners(gray_image, chessboard_shape)
    refined_corner_locations = cv.cornerSubPix(gray_image, corner_locations, half_of_window_size, zero_zone,termination_criteria)
    world_points_array.append(world_points)
    image_points_array_left.append(refined_corner_locations)
image = cv.imread(image_directory + "/CameraL00.bmp", cv.IMREAD_GRAYSCALE)
return_value, left_camera_matrix, left_distortion_coeficients, rotation_vectors, translation_vectors = cv.calibrateCamera(world_points_array, image_points_array_left, np.shape(image), None, None)
left_focal_length = left_camera_matrix[0,0]
print("focal_length: " , left_focal_length)
print("camera_matrix: " , left_camera_matrix)
print("distortion_coeficients: " , left_distortion_coeficients)
np.save("left_intrinsic_parameters_test.npy", left_camera_matrix)
np.save("left_distortion_parameters_test.npy", left_distortion_coeficients)

#right_camera
image_directory = os.getcwd() + "/Stereo_Calibration_Practice_Images/Practice/R"
data_path = os.path.join(image_directory,'*.bmp')
files = glob.glob(data_path)
number_of_inner_column_corners = 10
number_of_inner_column_rows = 7
chessboard_shape = (number_of_inner_column_corners,number_of_inner_column_rows)
zero_zone = (-1,-1)
half_of_window_size = (11,11)
termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
image_points_array_right = []
for file in files:
    gray_image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    return_value, corner_locations = cv.findChessboardCorners(gray_image, chessboard_shape)
    refined_corner_locations = cv.cornerSubPix(gray_image, corner_locations, half_of_window_size, zero_zone,termination_criteria)
    image_points_array_right.append(refined_corner_locations)
image = cv.imread(image_directory + "/CameraR00.bmp", cv.IMREAD_GRAYSCALE)
return_value, right_camera_matrix, right_distortion_coeficients, rotation_vectors, translation_vectors = cv.calibrateCamera(world_points_array, image_points_array_right, np.shape(image), None, None)
right_focal_length = right_camera_matrix[0,0]
print("focal_length: " , right_focal_length)
print("camera_matrix: " , right_camera_matrix)
print("distortion_coeficients: " , right_distortion_coeficients)
np.save("right_intrinsic_parameters_test.npy", right_camera_matrix)
np.save("right_distortion_parameters_test.npy", right_distortion_coeficients)

np.save("world_points_test.npy" , world_points_array)
np.save("image_points_left_test.npy" , image_points_array_left)
np.save("image_points_right_test.npy" , image_points_array_right)


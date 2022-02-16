import cv2 as cv
import numpy as np
import glob
import os

#task 2
# finds the calibration parameters for the camera test images

# Left Camera
image_directory = os.getcwd() + "/LeftCalibrationPics"
data_path = os.path.join(image_directory,'*.png')
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
left_image_points_array = []
for file in files:
    gray_image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    return_value, corner_locations = cv.findChessboardCorners(gray_image, chessboard_shape)
    pattern_was_found = False
    refined_corner_locations = cv.cornerSubPix(gray_image, corner_locations, half_of_window_size, zero_zone,termination_criteria)
    # cv.drawChessboardCorners(gray_image,chessboard_shape,refined_corner_locations,pattern_was_found)
    # cv.imshow("calibration_image", gray_image)
    # cv.waitKey(0)
    world_points_array.append(world_points)
    left_image_points_array.append(refined_corner_locations)
image = cv.imread(image_directory + "/0.png", cv.IMREAD_GRAYSCALE)
return_value, left_camera_matrix, left_distortion_coeficients, rotation_vectors, translation_vectors = cv.calibrateCamera(world_points_array, left_image_points_array, np.shape(image), None, None)
left_focal_length = left_camera_matrix[0,0]
print("Left focal_length: " , left_focal_length)
print("Left camera_matrix: " , left_camera_matrix)
print("Left distortion_coeficients: " , left_distortion_coeficients)

# right_camera
image_directory = os.getcwd() + "/RightCalibrationPics"
data_path = os.path.join(image_directory,'*.png')
files = glob.glob(data_path)
number_of_inner_column_corners = 10
number_of_inner_column_rows = 7
chessboard_shape = (number_of_inner_column_corners , number_of_inner_column_rows)
zero_zone = (-1 , -1)
half_of_window_size = (11,11)
termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
world_points = np.zeros((number_of_inner_column_corners*number_of_inner_column_rows,3), np.float32)
world_points[:,:2] = np.mgrid[0:number_of_inner_column_corners,0:number_of_inner_column_rows].T.reshape(-1,2)
world_points_array = []
right_image_points_array = []
for file in files:
    gray_image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    return_value, corner_locations = cv.findChessboardCorners(gray_image, chessboard_shape)
    # cv.drawChessboardCorners(gray_image,chessboard_shape,corner_locations,pattern_was_found)
    # cv.imshow("calibration_image", gray_image)
    # cv.waitKey(0)
    refined_corner_locations = cv.cornerSubPix(gray_image, corner_locations, half_of_window_size, zero_zone,termination_criteria)
    world_points_array.append(world_points)
    right_image_points_array.append(refined_corner_locations)
image = cv.imread(image_directory + "/0.png", cv.IMREAD_GRAYSCALE)
return_value, right_camera_matrix, right_distortion_coeficients, rotation_vectors, translation_vectors = cv.calibrateCamera(world_points_array, right_image_points_array, np.shape(image), None, None)
right_focal_length = right_camera_matrix[0,0]
print("Right focal_length: " , right_focal_length)
print("Right camera_matrix: " , right_camera_matrix)
print("Right distortion_coeficients: " , right_distortion_coeficients)

# np.save("right_intrinsic_parameters.npy", right_camera_matrix)
# np.save("right_distortion_parameters.npy", right_distortion_coeficients)
# np.save("left_intrinsic_parameters.npy", left_camera_matrix)
# np.save("left_distortion_parameters.npy", left_distortion_coeficients)

# np.save("world_points.npy" , world_points_array)
# np.save("left_image_points.npy" , left_image_points_array)
# np.save("right_image_points.npy" , right_image_points_array)



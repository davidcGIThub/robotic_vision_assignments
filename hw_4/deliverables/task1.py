import cv2 as cv
import numpy as np
import os

# task 1 3D Measurements

# Load parameters
right_camera_matrix = np.load("right_intrinsic_parameters.npy")
right_distortion_parameters = np.load("right_distortion_parameters.npy")
right_rectification = np.load("right_rectification.npy")
right_projection = np.load("right_projection.npy")
left_camera_matrix = np.load("left_intrinsic_parameters.npy")
left_distortion_parameters = np.load("left_distortion_parameters.npy")
left_rectification = np.load("left_rectification.npy")
left_projection = np.load("left_projection.npy")
disparity_depth_mapping = np.load("disparity_depth_mapping.npy")

# Define some parameters
number_of_inner_column_corners = 10
number_of_inner_column_rows = 7
chessboard_shape = ( number_of_inner_column_corners , number_of_inner_column_rows )
zero_zone = ( -1 , -1 )
half_of_window_size = ( 11 , 11 )
termination_criteria = ( cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER , 40 , 0.001 )

# Left Camera
image_directory = os.getcwd() + "/StereoCalibrationPics/L"
left_image = cv.imread(image_directory + "/0.png")
gray_left_image = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
return_value, left_corner_locations = cv.findChessboardCorners(gray_left_image, chessboard_shape)
refined_left_corner_locations = cv.cornerSubPix(gray_left_image, left_corner_locations, half_of_window_size, zero_zone,termination_criteria)
pattern_was_found = False
left_outer_points = np.array([refined_left_corner_locations[0] , refined_left_corner_locations[9], refined_left_corner_locations[60], refined_left_corner_locations[69]])
cv.drawChessboardCorners(left_image,chessboard_shape,left_outer_points,pattern_was_found)
cv.imshow("left_chessboard_corners", left_image)
cv.waitKey(0)
print("left_outer_points: " , left_outer_points)
print("shape: " , np.shape(left_outer_points))
print("type: " , type(left_outer_points))

# Right Camera
image_directory = os.getcwd() + "/StereoCalibrationPics/R"
right_image = cv.imread(image_directory + "/0.png")
gray_right_image = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
return_value, right_corner_locations = cv.findChessboardCorners(gray_right_image, chessboard_shape)
refined_right_corner_locations = cv.cornerSubPix(gray_right_image, right_corner_locations, half_of_window_size, zero_zone,termination_criteria)
right_outer_points = np.array([refined_right_corner_locations[0] , refined_right_corner_locations[9], refined_right_corner_locations[60], refined_right_corner_locations[69]])
cv.drawChessboardCorners(right_image,chessboard_shape,right_outer_points,pattern_was_found)
cv.imshow("right_chessboard_corners", right_image)
cv.waitKey(0)

#Undistort and rectify points

left_dst = cv.undistortPoints(left_outer_points, left_camera_matrix, left_distortion_parameters,R=left_rectification,P=left_projection) 
right_dst = cv.undistortPoints(right_outer_points, right_camera_matrix, right_distortion_parameters, R = right_rectification, P=right_projection)
print("left_dst: " , left_dst)
print("left_dst: " , left_dst[:,:,0].flatten())
left_x_points = left_dst[:,:,0].flatten()[:,None]
left_y_points = left_dst[:,:,1].flatten()[:,None]
disparity = (left_dst[:,:,0].flatten() - right_dst[:,:,0].flatten())[:,None]
left_input = np.concatenate((left_x_points,left_y_points,disparity),1).reshape((4,1,3))

right_x_points = right_dst[:,:,0].flatten()[:,None]
right_y_points = right_dst[:,:,1].flatten()[:,None]
right_input = np.concatenate((right_x_points,right_y_points,disparity),1).reshape((4,1,3))

# Perspective transform

left_frame_3d_points = cv.perspectiveTransform(left_input,disparity_depth_mapping)
right_frame_3d_points = cv.perspectiveTransform(right_input,disparity_depth_mapping)

print("left_frame_3d_points: " , left_frame_3d_points)
print("right_frame_3d_points: " , right_frame_3d_points)
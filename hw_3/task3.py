import cv2 as cv
import numpy as np
import os


# load images
left_image_path = os.getcwd() + "/StereoCalibrationPics/L/0.png"
right_image_path = os.getcwd() + "/StereoCalibrationPics/R/0.png"
left_image = cv.imread(left_image_path)
right_image = cv.imread(right_image_path)
# load parameters
right_camera_matrix = np.load("right_intrinsic_parameters.npy")
right_distortion_parameters = np.load("right_distortion_parameters.npy")
left_camera_matrix = np.load("left_intrinsic_parameters.npy")
left_distortion_parameters = np.load("left_distortion_parameters.npy")
fundamental_matrix = np.load("fundamental_matrix.npy")
# undistort the images
corrected_left_image = cv.undistort(left_image,left_camera_matrix,left_distortion_parameters,None)
gray_corrected_left_image = cv.cvtColor(corrected_left_image, cv.COLOR_BGR2GRAY)
corrected_right_image = cv.undistort(right_image,right_camera_matrix,right_distortion_parameters,None)
gray_corrected_right_image = cv.cvtColor(corrected_right_image, cv.COLOR_BGR2GRAY)
# select points on the image
number_of_inner_column_corners = 10
number_of_inner_column_rows = 7
half_of_window_size = (11,11)
zero_zone = (-1,-1)
termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
chessboard_shape = (number_of_inner_column_corners , number_of_inner_column_rows)
return_value, left_corner_locations = cv.findChessboardCorners(gray_corrected_left_image, chessboard_shape)
left_corner_locations = cv.cornerSubPix(gray_corrected_left_image, left_corner_locations, half_of_window_size, zero_zone,termination_criteria)
return_value, right_corner_locations = cv.findChessboardCorners(gray_corrected_right_image, chessboard_shape)
right_corner_locations = cv.cornerSubPix(gray_corrected_right_image, right_corner_locations, half_of_window_size, zero_zone,termination_criteria)
point_indices = np.array([0,18,34])
selected_left_image_points = left_corner_locations[point_indices]
selected_right_image_points = right_corner_locations[point_indices]
# draw the points
pattern_was_found = False
cv.drawChessboardCorners(corrected_left_image,chessboard_shape,selected_left_image_points,pattern_was_found)
cv.imshow("Left Image", corrected_left_image)
cv.waitKey(0)
cv.drawChessboardCorners(corrected_right_image,chessboard_shape,selected_right_image_points,pattern_was_found)
cv.imshow("Right Image", corrected_right_image)
cv.waitKey(0)

def draw_epipolar_lines(image, lines):
    number_of_lines = np.shape(lines)[0]
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    columns_in_image = np.shape(gray_image)[1]
    color = (0,0,255)
    print("lines: " , lines)
    for i in range(number_of_lines):
        x0 = 0
        y0 = int(-lines[i,2]/lines[i,1])
        x1 = columns_in_image
        y1 = int(-(lines[i,2]+lines[i,0]*columns_in_image)/lines[i,1])
        image = cv.line(image, (x0,y0), (x1,y1), color , 1)
    return image

right_lines = cv.computeCorrespondEpilines(selected_left_image_points,1, fundamental_matrix).reshape(-1,3)
right_image_with_lines = draw_epipolar_lines(corrected_right_image, right_lines)

left_lines = cv.computeCorrespondEpilines(selected_right_image_points,2, fundamental_matrix).reshape(-1,3)
left_image_with_lines = draw_epipolar_lines(corrected_left_image, left_lines)

cv.imshow("right_image_with_lines", right_image_with_lines)
cv.waitKey(0)

cv.imshow("left_image_with_lines", left_image_with_lines)
cv.waitKey(0)
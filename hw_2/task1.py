import cv2 as cv
import os

#task 1

calibration_images_folder_path = os.getcwd() + "/Calibration_Images_JPG"
image = cv.imread(calibration_images_folder_path + "/AR1.jpg")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
number_of_inner_column_corners = 10
number_of_inner_column_rows = 7
chessboard_shape = (number_of_inner_column_corners,number_of_inner_column_rows)
return_value, corner_locations = cv.findChessboardCorners(gray_image, chessboard_shape)
zero_zone = (-1,-1)
half_of_window_size = (10,10)
termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
refined_corner_locations = cv.cornerSubPix(gray_image, corner_locations, half_of_window_size, zero_zone,termination_criteria)
pattern_was_found = False
cv.drawChessboardCorners(image,chessboard_shape,refined_corner_locations,pattern_was_found)
cv.imshow("chessboard_corners", image)
cv.waitKey(0)

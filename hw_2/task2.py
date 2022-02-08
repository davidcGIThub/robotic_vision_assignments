import cv2 as cv
import numpy as np
import glob
import os

#task 2

image_directory = os.getcwd() + "/Calibration_Images_JPG"
data_path = os.path.join(image_directory,'*.jpg') 
files = glob.glob(data_path)
number_of_inner_column_corners = 10 
number_of_inner_column_rows = 7
chessboard_shape = (number_of_inner_column_corners,number_of_inner_column_rows)
zero_zone = (-1,-1)
half_of_window_size = (11,11)
termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
world_points = np.zeros((number_of_inner_column_corners*number_of_inner_column_rows,3), np.float32)
world_points[:,:2] = np.mgrid[0:number_of_inner_column_corners,0:number_of_inner_column_rows].T.reshape(-1,2)
print("world_points: " , np.shape(world_points))
world_points_array = []
image_points_array = []
for file in files:
    gray_image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    return_value, corner_locations = cv.findChessboardCorners(gray_image, chessboard_shape)
    refined_corner_locations = cv.cornerSubPix(gray_image, corner_locations, half_of_window_size, zero_zone,termination_criteria)
    print("refined_corner_locations: " , np.shape(refined_corner_locations))
    world_points_array.append(world_points)
    image_points_array.append(refined_corner_locations)
    # pattern_was_found = False
    # cv.drawChessboardCorners(gray_image,chessboard_shape,refined_corner_locations,pattern_was_found)
    # cv.imshow("calibration_image", gray_image)
    # cv.waitKey(0)
image = cv.imread(image_directory + "/AR1.jpg", cv.IMREAD_GRAYSCALE)
return_value, camera_matrix, distortion_coeficients, rotation_vectors, translation_vectors = cv.calibrateCamera(world_points_array, image_points_array, np.shape(image), None, None)
pixel_size = 0.0074 # mm
focal_length = pixel_size * camera_matrix[0,0]
print("focal_length: " , focal_length)
print("camera_matrix: " , camera_matrix)
print("distortion_coeficients: " , distortion_coeficients)
np. save("intrinsic_parameters.npy", camera_matrix)
np. save("distortion_parameters.npy", distortion_coeficients)


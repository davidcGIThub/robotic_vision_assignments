import cv2 as cv
import numpy as np
import os


left_image_path = os.getcwd() + "/StereoCalibrationPics/L/0.png"
right_image_path = os.getcwd() + "/StereoCalibrationPics/R/0.png"
left_image = cv.imread(left_image_path)
right_image = cv.imread(right_image_path)

right_camera_matrix = np.load("right_intrinsic_parameters.npy")
right_distortion_parameters = np.load("right_distortion_parameters.npy")
left_camera_matrix = np.load("left_intrinsic_parameters.npy")
left_distortion_parameters = np.load("left_distortion_parameters.npy")
fundamental_matrix = np.load("fundamental_matrix.npy")

corrected_left_image = cv.undistort(left_image,left_camera_matrix,left_distortion_parameters,None)
corrected_right_image = cv.undistort(right_image,right_camera_matrix,right_distortion_parameters,None)
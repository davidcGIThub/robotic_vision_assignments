import cv2 as cv
import numpy as np
import os

current_directory = os.getcwd()
image_directory = os.getcwd() + "/checkerboard_image_home"
distortion_parameters = np.load("distortion_parameters_home.npy")
camera_matrix = np.load("intrinsic_parameters_home.npy")
image = cv.imread(image_directory + "/checkerboard5.jpg")
corrected_image = cv.undistort(image,camera_matrix,distortion_parameters,None)
difference_image = cv.absdiff(image , corrected_image)
cv.imshow("difference_image" , difference_image)
cv.imwrite("difference_image.jpg", difference_image)
cv.waitKey(0)
 
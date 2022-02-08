import cv2 as cv
import numpy as np
import os

current_directory = os.getcwd()
image_directory = os.getcwd() + "/Distortion_Correction_Images"
distortion_parameters = np.load("distortion_parameters.npy")
camera_matrix = np.load("intrinsic_parameters.npy")
close_image = cv.imread(image_directory + "/Close.jpg" )
far_image = cv.imread(image_directory + "/Far.jpg" )
turned_image = cv.imread(image_directory + "/Turned.jpg" )
corrected_far_image = cv.undistort(far_image,camera_matrix,distortion_parameters,None)
corrected_close_image = cv.undistort(close_image,camera_matrix,distortion_parameters,None)
corrected_turned_image = cv.undistort(turned_image,camera_matrix,distortion_parameters,None)
close_difference_image = cv.absdiff(close_image,corrected_close_image)
far_difference_image =  cv.absdiff(far_image ,corrected_far_image)
turned_difference_image = cv.absdiff(turned_image , corrected_turned_image)

cv.imshow("close_difference_image" , close_difference_image)
cv.imwrite("close_difference_image.jpg", close_difference_image)
cv.waitKey(0)

cv.imshow("far_difference_image" , far_difference_image)
cv.imwrite("far_difference_image.jpg", far_difference_image)
cv.waitKey(0)

cv.imshow("turned_difference_image" , turned_difference_image)
cv.imwrite("turned_difference_image.jpg", turned_difference_image)
cv.waitKey(0)
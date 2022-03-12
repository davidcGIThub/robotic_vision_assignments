from hashlib import new
import numpy as np
import cv2 as cv
import glob
import os
import copy
  
image_directory = os.getcwd() + "/Sequence"
original_gray_left_image = cv.imread(image_directory + "/L/27.png", cv.IMREAD_GRAYSCALE)
gray_left_image = original_gray_left_image 
original_gray_right_image = cv.imread(image_directory + "/R/27.png", cv.IMREAD_GRAYSCALE)
gray_right_image = original_gray_right_image    
initial_ball_spot = np.array([None,None])
initial_window_size = 40

# def click_event(event, x, y, flags, params):
#     global initial_ball_location
#     if event == cv.EVENT_LBUTTONDOWN:
#         initial_ball_location = np.array([x,y])

# cv.imshow('gray_left_image', gray_left_image)
# cv.setMouseCallback('gray_left_image', click_event)
# cv.waitKey(0)
# left_initial_ball_location = initial_ball_location
left_initial_ball_location = np.array([364, 102])

# cv.imshow('gray_right image', gray_right_image)
# cv.setMouseCallback('gray_right image', click_event)
# cv.waitKey(0)
# right_initial_ball_location = initial_ball_location
right_initial_ball_location = np.array([276, 102])

# find ball pixel location on left and right images

# put roi on picture
window_size = initial_window_size

#look through left images
data_path = os.path.join(image_directory + "/L" ,'*.png')
files = sorted(glob.glob(data_path))
left_ball_location = left_initial_ball_location
image_number = -1
left_points = []

for file in files:
    left_image = cv.imread(file)
    gray_left_image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    diff_thresh = 10
    # static_difference_image = cv.absdiff(new_gray_left_image,original_gray_left_image)
    difference_image = cv.absdiff(original_gray_left_image,gray_left_image)
    difference_image[difference_image > diff_thresh] = gray_left_image[difference_image > diff_thresh]
    blur_kernel_size = 13
    blur_thresh = 10   
    noise_removed_image = cv.medianBlur(difference_image, blur_kernel_size) 
    noise_removed_image[noise_removed_image < blur_thresh] = 0
    max_val = np.amax(noise_removed_image)+1
    noise_removed_image = (noise_removed_image*255.0/max_val).astype(np.uint8)
    width = np.shape(gray_left_image)[0]
    height = np.shape(gray_left_image)[1]
    left = np.clip(left_ball_location[1]-window_size,0,width-1)
    right = np.clip(left_ball_location[1]+window_size,0,width-1)
    top = np.clip(left_ball_location[0]-window_size,0,height-1)
    bottom = np.clip(left_ball_location[0]+window_size,0,height-1)
    roi_image = gray_left_image[left:right , top:bottom]
    minDist = 500 # minimum distance between the cetners for the detected circles
    dp = 1.25 # inverse ratio fo the accumulator resolution
    circles = cv.HoughCircles(noise_removed_image, cv.HOUGH_GRADIENT, dp, minDist, param1=140, param2=15, minRadius=1, maxRadius=35)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x = circles[0,0]
        y = circles[0,1]
        r = circles[0,2]
        color = (0, 255, 0)
        thickness = 3  
        cv.circle(left_image, (x, y), r, color, thickness)
        left_ball_location = [x,y]
        window_size = r*5 
        image_number += 1
    square_thickness = 3
    square_color = (255, 0, 0)
    cv.rectangle(left_image,[top,left],[bottom,right],square_color,square_thickness)
    cv.imshow("left_image",left_image)
    cv.waitKey(0)
    if image_number%5 == 0 and image_number < 26:
        cv.imwrite(str(image_number) + "_left_image.png" , left_image)
        left_points.append(left_ball_location)
    
data_path = os.path.join(image_directory + "/R/" ,'*.png')
files = sorted(glob.glob(data_path))
right_ball_location = right_initial_ball_location
image_number = -1
window_size = initial_window_size
right_points = []
for file in files:
    right_image = cv.imread(file)
    gray_right_image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    diff_thresh = 10
    # static_difference_image = cv.absdiff(new_gray_right_image,original_gray_right_image)
    difference_image = cv.absdiff(original_gray_right_image,gray_right_image)
    difference_image[difference_image > diff_thresh] = gray_right_image[difference_image > diff_thresh]
    blur_kernel_size = 13
    blur_thresh = 10   
    noise_removed_image = cv.medianBlur(difference_image, blur_kernel_size) 
    noise_removed_image[noise_removed_image < blur_thresh] = 0
    max_val = np.amax(noise_removed_image)+1
    noise_removed_image = (noise_removed_image*255.0/max_val).astype(np.uint8)
    width = np.shape(gray_right_image)[0]
    height = np.shape(gray_right_image)[1]
    left = np.clip(right_ball_location[1]-window_size,0,width-1)
    right = np.clip(right_ball_location[1]+window_size,0,width-1)
    top = np.clip(right_ball_location[0]-window_size,0,height-1)
    bottom = np.clip(right_ball_location[0]+window_size,0,height-1)
    # roi_image = gray_right_image[left:right , top:bottom] 
    minDist = 500 # minimum distance between the cetners for the detected circles
    dp = 1.25 # inverse ratio fo the accumulator resolution
    circles = cv.HoughCircles(noise_removed_image, cv.HOUGH_GRADIENT, dp, minDist, param1=140, param2=15, minRadius=1, maxRadius=35)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x = circles[0,0]
        y = circles[0,1]
        r = circles[0,2]
        color = (0, 255, 0)
        thickness = 3  
        cv.circle(right_image, (x, y), r, color, thickness)
        right_ball_location = [x,y]
        window_size = r*5 
        image_number += 1
    square_thickness = 3
    square_color = (255, 0, 0)
    cv.rectangle(right_image,[top,left],[bottom,right],square_color,square_thickness)
    cv.imshow("right_image",right_image)
    cv.waitKey(0)
    if image_number%5 == 0 and image_number < 26:
        cv.imwrite(str(image_number) + "_right_image.png" , right_image)
        right_points.append(right_ball_location)

print("right_points: " , np.array(right_points))
print("left_points: " , np.array(left_points))
np.save("right_points.npy" , np.array(right_points))
np.save("left_points.npy" , np.array(left_points))
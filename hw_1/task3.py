import cv2
import numpy as np
import glob
import copy

img_array = []
filenames = [img for img in glob.glob('/home/david/Code/robotic_vision_assignments/hw_1/Baseball_Practice_Images/*.png')]
filenames.sort() # ADD THIS LINE
buffer = cv2.imread('/home/david/Code/robotic_vision_assignments/hw_1/Baseball_Practice_Images/L40.png', cv2.IMREAD_GRAYSCALE)
for filename in filenames:
    original_image = cv2.imread(filename)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    difference_image = np.zeros(buffer.shape,dtype=np.uint8)
    cv2.absdiff(buffer, gray_image, difference_image)
    ret,threshhold_image = cv2.threshold(difference_image,2,255,cv2.THRESH_BINARY)
    noise_removed_image = cv2.medianBlur(threshhold_image, 21)
    masked = cv2.bitwise_and(gray_image, gray_image, mask=noise_removed_image)
    modified_image = copy.deepcopy(original_image)
    # Video editing
    dp = 1.3 # inverse ratio fo the accumulator resolution
    minDist = 50 # minimum distance between the cetners fo the detected circles
    # circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp, minDist)
    circles = cv2.HoughCircles(masked, cv2.HOUGH_GRADIENT, dp, minDist, param1=130, param2=30, minRadius=0, maxRadius=25)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            color = (0, 255, 0)
            thickness = 4
            cv2.circle(modified_image, (x, y), r, color, thickness)
    img_array.append(modified_image)
    # img_array.append(cv2.cvtColor(noise_removed_image, cv2.COLOR_GRAY2BGR))
    buffer = gray_image

height, width, layers = img_array[0].shape
size = (width,height)
out = cv2.VideoWriter('/task3.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
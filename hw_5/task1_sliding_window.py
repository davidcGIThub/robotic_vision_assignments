from cv2 import cvtColor
from matplotlib.pyplot import gray
import numpy as np
import cv2 as cv


# Open video
video = cv.VideoCapture('tools.MOV')
if (video.isOpened()== False):
  print("Error opening video stream or file")

# Prepere first several frames to skip
number_of_frames_to_skip = 5
image_array = []
while len(image_array) < number_of_frames_to_skip:
  ret, image = video.read()
  if ret == True:
    image_array.append(image)
    
# Parameters for good features to track
number_of_tracks = 300
quality_level = 0.1
max_corners = 300
min_distance = 10
block_size =  7
# Parameters for optical flow calculation
window_size = (15, 15)
max_level = 3
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,30, 0.03)
# Color parameters
line_color = (0,255,0)
point_color = (0,0,255)

# Prepare first several sets of points
points_array = []
image = cv.cvtColor(image_array[0],cv.COLOR_BGR2GRAY)
for i in range(number_of_frames_to_skip-1):
  if i == 0:
    points = cv.goodFeaturesToTrack(image,maxCorners=max_corners, qualityLevel = quality_level, minDistance= min_distance, blockSize=block_size)
  else:
    next_image = cv.cvtColor(image_array[i+1],cv.COLOR_BGR2GRAY)
    points = cv.calcOpticalFlowPyrLK(prevImg= image,
                            nextImg= next_image,prevPts= points_array[0],nextPts=None,
                            maxLevel=max_corners, winSize=window_size, criteria=criteria)
    if len(points) < number_of_tracks:
      points = cv.goodFeaturesToTrack(next_image,maxCorners=max_corners, qualityLevel = quality_level, minDistance= min_distance, blockSize=block_size)
  points_array.append(points)

# loop through video
while(video.isOpened()):
  ret, image = video.read()
  gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  if ret == True:
    # calculate the next points
    previous_points = points_array[0]
    previous_gray_image = cv.cvtColor(image_array[0],cv.COLOR_BGR2GRAY)
    next_points , status , error = cv.calcOpticalFlowPyrLK(prevImg= previous_gray_image,
                            nextImg= gray_image,prevPts= previous_points,nextPts=None,
                            maxLevel=max_corners, winSize=window_size, criteria=criteria )

    # Choose the good points
    next_points_good = next_points[status == 1]
    previous_points_good = previous_points[status == 1]
    
    # Draw the tracks
    for i in range(len(next_points_good)):
      x_new = int(next_points_good[i,0])
      y_new = int(next_points_good[i,1])
      x_old = int(previous_points_good[i,0])
      y_old = int(previous_points_good[i,1])
      image = cv.line(image, (int(x_new), int(y_new)), (int(x_old), int(y_old)),
                        line_color, 2)    
      image = cv.circle(image, (int(x_new), int(y_new)), 5, point_color, -1)
    cv.imshow('Frame',image)

    # Update image array
    image_array.pop(0)
    image_array.append(image)

    # Update the points array
    points_array.pop(0)
    if len(next_points_good) < number_of_tracks:
      new_points = cv.goodFeaturesToTrack(gray_image,maxCorners=max_corners, qualityLevel = quality_level, minDistance= min_distance, blockSize=block_size)
    else:
        points = next_points_good.reshape(-1, 1, 2)
    points_array.append(points)

    # break key
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break
 
video.release()
cv.destroyAllWindows()

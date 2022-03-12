from cv2 import cvtColor
from matplotlib.pyplot import gray
import numpy as np
import cv2 as cv

# video = cv.VideoCapture('soda.MOV')
video = cv.VideoCapture('tools.MOV')

if (video.isOpened()== False):
  print("Error opening video stream or file")
number_of_frames_to_skip = 6
image_array = []
while len(image_array) < number_of_frames_to_skip:
  ret, image = video.read()
  if ret == True:
    image_array.append(image)

frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
video_result = cv.VideoWriter('optical_flow.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         10, size)


quality_level = 0.1
max_corners = 300
min_distance = 10
block_size =  7
previous_gray_image = cv.cvtColor(image_array[0],cv.COLOR_BGR2GRAY)
previous_points = cv.goodFeaturesToTrack(previous_gray_image,maxCorners=max_corners, qualityLevel = quality_level, minDistance= min_distance, blockSize=block_size)
line_color = (0,255,0)
point_color = (0,0,255)
window_size = (15, 15)
max_level = 3
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,30, 0.03)
current_frame = 0
while(video.isOpened()):
  ret, image = video.read()
  gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  if ret == True:
    if current_frame%number_of_frames_to_skip == 0:
      next_points , status , error = cv.calcOpticalFlowPyrLK(prevImg= previous_gray_image,
                              nextImg= gray_image,prevPts= previous_points,nextPts=None,
                              maxLevel=max_corners, winSize=window_size, criteria=criteria )
      new_points_good = next_points[status == 1]
      old_points_good = previous_points[status == 1]
      previous_gray_image = gray_image.copy()
      # previous_points = next_points
      if len(new_points_good) < 300:
        previous_points = cv.goodFeaturesToTrack(previous_gray_image,maxCorners=max_corners, qualityLevel = quality_level, minDistance= min_distance, blockSize=block_size)
      else:
        previous_points = new_points_good.reshape(-1, 1, 2)
      # draw the tracks
      for i in range(len(new_points_good)):
        x_new = int(new_points_good[i,0])
        y_new = int(new_points_good[i,1])
        x_old = int(old_points_good[i,0])
        y_old = int(old_points_good[i,1])
        image = cv.line(image, (int(x_new), int(y_new)), (int(x_old), int(y_old)),
                          line_color, 2)    
        image = cv.circle(image, (int(x_new), int(y_new)), 5, point_color, -1)
      cv.imshow('Frame',image)
      video_result.write(image)
    current_frame += 1
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break
 
video.release()
cv.destroyAllWindows()

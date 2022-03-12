import numpy as np
import cv2
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H_%M")

# This will return video from the first webcam on your computer.
camera_capture = cv2.VideoCapture(0)  
  
# Define the codec and create VideoWriter object, FourCC is a 4-byte code used to specify the video codec. 
# codes can be obtained at https://www.fourcc.org/codecs.php
# Xvid is a free video codec following the MPEG-4 video coding standard.
fourcc_code = cv2.VideoWriter_fourcc(*'XVID')

# creating an output video writer object, parameters are (filename, fourcc code, fps, frame size)
output_video = cv2.VideoWriter(str(current_time) + 'catcher_video.avi', fourcc_code, 20.0, (640, 480))
  
isFrameCaptured = True
isFrameCaptured, frame = camera_capture.read()
delay_time = 10
threshhold = 60
delay = delay_time
# loop runs after capturing has been initialized while while frames are being captured. 
while(isFrameCaptured):
    # Wait for 'a' key to stop the program 
    # reads frames from a camera, and checks if frames are being captured
    isFrameCaptured, new_frame = camera_capture.read() 
    # The original input frame is shown in the window 
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    new_gray_frame = cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)
    difference_image = cv2.absdiff(gray_frame,new_gray_frame)
    if np.any(difference_image[difference_image>threshhold]) or delay > 0:
        print("writing")
        output_video.write(new_frame)
        if np.any(difference_image[difference_image>threshhold]):
            delay = 10
    if delay > 0:
        delay -= 1
    frame = new_frame
    cv2.imshow('Original', new_frame)

      
    # Wait for 'a' key to stop the program 
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
  
# Close the window / Release webcam
camera_capture.release()
  
# After we release our webcam, we also release the output
output_video.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()
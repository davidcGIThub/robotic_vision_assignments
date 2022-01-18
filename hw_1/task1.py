import numpy as np
import cv2

# This will return video from the first webcam on your computer.
camera_capture = cv2.VideoCapture(0)  
  
# Define the codec and create VideoWriter object, FourCC is a 4-byte code used to specify the video codec. 
# codes can be obtained at https://www.fourcc.org/codecs.php
# Xvid is a free video codec following the MPEG-4 video coding standard.
fourcc_code = cv2.VideoWriter_fourcc(*'XVID')

# creating an output video writer object, parameters are (filename, fourcc code, fps, frame size)
output_video = cv2.VideoWriter('task1.avi', fourcc_code, 20.0, (640, 480))
  
isFrameCaptured = True

# loop runs after capturing has been initialized while while frames are being captured. 
while(isFrameCaptured):
    # reads frames from a camera, and checks if frames are being captured
    isFrameCaptured, frame = camera_capture.read() 
      
    # The original input frame is shown in the window 
    cv2.imshow('Original', frame)
    output_video.write(frame)
      
    # Wait for 'a' key to stop the program 
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
  
# Close the window / Release webcam
camera_capture.release()
  
# After we release our webcam, we also release the output
output_video.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()
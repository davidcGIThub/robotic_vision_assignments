import cv2 as cv
import numpy as np
import os

# load images
left_image_path = os.getcwd() + "/StereoCalibrationPics/L/0.png"
right_image_path = os.getcwd() + "/StereoCalibrationPics/R/0.png"
left_image = cv.imread(left_image_path)
right_image = cv.imread(right_image_path)
gray_left_image = cv.cvtColor(left_image, cv.COLOR_RGB2GRAY)
gray_right_image = cv.cvtColor(right_image, cv.COLOR_RGB2GRAY)
image_size = np.shape(gray_left_image)

cv.imshow("Left Image", left_image)
cv.waitKey(0)     
cv.imshow("Right Image", right_image)
cv.waitKey(0)           

# load parameters
right_camera_matrix = np.load("right_intrinsic_parameters.npy")
right_distortion_parameters = np.load("right_distortion_parameters.npy")
left_camera_matrix = np.load("left_intrinsic_parameters.npy")
left_distortion_parameters = np.load("left_distortion_parameters.npy")
fundamental_matrix = np.load("fundamental_matrix.npy")
rotation_matrix = np.load("rotation_between_cameras.npy")
translation_matrix = np.load("translation_between_cameras.npy")
print("fundamental: " , fundamental_matrix) 
print("rotation_matrix: " , rotation_matrix) 
print("translation_matrix: " , translation_matrix) 
s = cv.stereoRectify(left_camera_matrix, left_distortion_parameters, 
                    right_camera_matrix, right_distortion_parameters, 
                    image_size, rotation_matrix, translation_matrix)
left_rectification = s[0]
right_rectification = s[1]
left_projection = s[2]
right_projection = s[3]
disparity_depth_mapping = s[4]
print("disparity_depth_mapping: " , disparity_depth_mapping)
roi_left = s[5]
roi_right = s[6]

np.save("right_rectification.npy", right_rectification)
np.save("right_projection.npy", right_projection)
np.save("left_rectification.npy" , left_rectification)
np.save("left_projection.npy", left_projection)
np.save("disparity_depth_mapping.npy", disparity_depth_mapping)
print("roi_left: " , roi_left)
print("roi_right: " , roi_right)


map1_x, map1_y = cv.initUndistortRectifyMap(left_camera_matrix, left_distortion_parameters,
                                            left_rectification, left_projection,
                                            image_size, cv.CV_16SC2)
map2_x, map2_y = cv.initUndistortRectifyMap(right_camera_matrix, right_distortion_parameters,
                                            right_rectification, right_projection,
                                            image_size, cv.CV_16SC2)

rectified_left_image = cv.remap(left_image, map1_x, map1_y,
                      interpolation=cv.INTER_NEAREST,
                      borderMode=cv.BORDER_CONSTANT,
                      borderValue=(0, 0, 0, 0))[0:477,2:480,:]

rectified_right_image = cv.remap(right_image, map2_x, map2_y,
                      interpolation=cv.INTER_NEAREST,
                      borderMode=cv.BORDER_CONSTANT,
                      borderValue=(0, 0, 0, 0))[7:473,0:480,:]   

end_column = image_size[1]-1
end_row = image_size[0]-1
x_points = np.array([[0,end_column],[0,end_column],[0,end_column]])
y_points = np.array([[int(end_row/3),int(end_row/3)],[int(end_row/1.1),int(end_row/1.1)],[int(end_row/5),int(end_row/5)]])

def draw_horizontal_lines(image, x_points, y_points):
    number_of_lines = np.shape(x_points)[0]
    color = (0,0,255)
    new_image = image.copy()
    for i in range(number_of_lines):
        x0 = x_points[i,0]
        y0 = y_points[i,0]
        x1 = x_points[i,1]
        y1 = y_points[i,1]
        new_image = cv.line(new_image, (x0,y0), (x1,y1), color , 1)
    return new_image

rectified_left_image_with_lines = draw_horizontal_lines(rectified_left_image, x_points, y_points)
cv.imshow("Rectified Left Image With Lines", rectified_left_image_with_lines)
cv.waitKey(0)   

rectified_right_image_with_lines = draw_horizontal_lines(rectified_right_image, x_points, y_points)
cv.imshow("Rectified Right Image With Lines", rectified_right_image_with_lines)
cv.waitKey(0)   

left_image_cropped = left_image[0:477,2:480,:]
right_image_cropped = right_image[7:473,0:480,:]

left_difference_image = cv.absdiff(left_image_cropped, rectified_left_image)
cv.imshow("Difference Image", left_difference_image )
cv.waitKey(0)

right_difference_image = cv.absdiff(right_image_cropped, rectified_right_image)
cv.imshow("Difference Image", right_difference_image)
cv.waitKey(0) 
import numpy as np
import cv2 as cv
import os

image_path = os.getcwd() + "/Object_Image_And_Data_For_Pose_Estimation/Object_with_Corners.jpg"
file_path = os.getcwd() + "/Object_Image_And_Data_For_Pose_Estimation/fileDownload.txt"
image_points = []
world_points = []
with open(file_path) as file:
    for line in file:
        line_data = np.fromstring(line, dtype=float, sep=' ').tolist()
        if len(line_data) == 2:
            image_points.append([line_data])
        if len(line_data) == 3:
            world_points.append(line_data)
image = cv.imread(image_path)
camera_matrix = np.load("intrinsic_parameters.npy")
distortion_parameters = np.load("distortion_parameters.npy")
return_value, rotation_vector, translation_vector = cv.solvePnP(np.array(world_points),np.array(image_points),camera_matrix,distortion_parameters)
rotation_matrix, jacobian = cv.Rodrigues(rotation_vector.flatten())
print("rotation_matrix: " , rotation_matrix)
print("translation_vector: " , translation_vector)

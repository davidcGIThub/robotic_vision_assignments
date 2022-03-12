import numpy as np
import cv2 as cv
from trajectory_predictor import TrajectoryPredictor

# Load parameters
right_camera_matrix = np.load("right_intrinsic_parameters.npy")
right_distortion_parameters = np.load("right_distortion_parameters.npy")
right_rectification = np.load("right_rectification.npy")
right_projection = np.load("right_projection.npy")
left_camera_matrix = np.load("left_intrinsic_parameters.npy")
left_distortion_parameters = np.load("left_distortion_parameters.npy")
left_rectification = np.load("left_rectification.npy")
left_projection = np.load("left_projection.npy")
disparity_depth_mapping = np.load("disparity_depth_mapping.npy")
left_points = np.reshape(np.load("right_points.npy"),(6,1,2)).astype(float)
right_points = np.reshape(np.load("left_points.npy"),(6,1,2)).astype(float)

left_dst = cv.undistortPoints(left_points, left_camera_matrix, left_distortion_parameters,R=left_rectification,P=left_projection) 
right_dst = cv.undistortPoints(right_points, right_camera_matrix, right_distortion_parameters, R = right_rectification, P=right_projection)
left_x_points = left_dst[:,:,0].flatten()[:,None]
left_y_points = left_dst[:,:,1].flatten()[:,None]
disparity = (left_dst[:,:,0].flatten() - right_dst[:,:,0].flatten())[:,None]
left_input = np.concatenate((left_x_points,left_y_points,disparity),1).reshape((6,1,3))

right_x_points = right_dst[:,:,0].flatten()[:,None]
right_y_points = right_dst[:,:,1].flatten()[:,None]
right_input = np.concatenate((right_x_points,right_y_points,disparity),1).reshape((6,1,3))

left_frame_3d_points = cv.perspectiveTransform(left_input,disparity_depth_mapping)
print("left_frame_3d_points: " , left_frame_3d_points)

transformation = np.array([[1 ,  0 ,  0 , 10],
                           [0 , 1 ,  0 , 0],
                           [0 ,  0 , 1 , 0],
                           [0 ,  0 ,  0 , 1]])
traj_pred = TrajectoryPredictor(transformation,2)
target_zone = traj_pred.get_target_zone_in_catcher_frame(np.reshape(left_frame_3d_points,(6,3)))
traj_pred.plot_trajectory_in_catcher_frame(np.reshape(left_frame_3d_points,(6,3)), 100)
print("target_zone: " , target_zone)

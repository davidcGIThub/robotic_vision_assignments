import numpy as np
from trajectory_predictor import  TrajectoryPredictor

transformation = np.array([[1 , 0 , 0 , 10],
                           [0 , -1 , 0 , 50],
                           [0 , 0 , -1 , 0],
                           [0 , 0 , 0 , 1]])
traj_pred = TrajectoryPredictor(transformation,2)
x_y_z_points_camera_frame = np.array([[0   , -3 , 22],
                                      [0.5 , -2 , 15],
                                      [2   , -1 , 10],
                                      [2.1 ,  0 , 7],
                                      [3   ,  1 , 3]])
transformed_points = traj_pred.get_transformed_points(x_y_z_points_camera_frame)
target_zone = traj_pred.get_target_zone_in_catcher_frame(x_y_z_points_camera_frame)

print("transformed_points: " , transformed_points)
print("target_zone: " , target_zone)

traj_pred.plot_trajectory_in_catcher_frame(x_y_z_points_camera_frame, 100)

import numpy as np
import matplotlib.pyplot as plt

class TrajectoryPredictor:
    def __init__(self, camera_to_catcher_transformation, degree = 2):
        """
        Constructor, needs the camera to catcher transformation in SE3 (4x4 matrix)
        """
        self._camera_to_catcher_transformation = camera_to_catcher_transformation
        self._degree = degree

    def get_target_zone_in_catcher_frame(self, points):
        """
        Returns the x, y location of where the ball will pass
        through the catcher plane in the catcher coordinate frame
        """
        transformed_points = self.__transform_points_to_catcher_frame(points)
        x_trajectory, y_trajectory = self.__calculate_ball_trajectory_with_polynomial_fit(transformed_points)
        intersection_point = self.__find_intersection_of_trajectory_and_catcher_plane(x_trajectory,y_trajectory)
        return intersection_point

    def get_transformed_points(self, points):
        transformed_points = self.__transform_points_to_catcher_frame(points)
        return transformed_points

    def plot_trajectory_in_catcher_frame(self, points, number_of_points_to_plot):
        """
        Plots an aeral and side view of the trajecory in the catcher 
        frame and its's points, as well as the location of the camera.
        """
        camera_position = self.__calculate_camera_position_in_catcher_frame()
        trajectory_points = self.__get_x_y_z_points_to_plot(points, number_of_points_to_plot)
        data_points = self.__transform_points_to_catcher_frame(points)
        plt.subplot(121)
        plt.gca().invert_xaxis()
        plt.title("Ball Trajectory")
        plt.scatter(data_points[:,2],data_points[:,0],color='r')
        plt.plot(trajectory_points[:,2],trajectory_points[:,0],color='r')
        plt.ylabel("x position")
        plt.xlabel("z position")
        plt.axis('equal')
        plt.scatter(camera_position[2],camera_position[0],color='b')
        plt.legend(["data points", "estimated trajectory" , "camera position"])
        plt.subplot(122)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.scatter(data_points[:,2],data_points[:,1],color='r')
        plt.plot(trajectory_points[:,2],trajectory_points[:,1],color='r')
        plt.ylabel("y position")
        plt.xlabel("z position")
        plt.axis('equal')
        plt.scatter(camera_position[2],camera_position[1],color='b')
        plt.legend(["data points", "estimated trajectory" , "camera position"])
        plt.show()

    def __get_x_y_z_points_to_plot(self, points, number_of_points_to_plot):
        transformed_points = self.__transform_points_to_catcher_frame(points)
        furthest_distance_from_catcher_index = np.argmax(np.abs(transformed_points[:,2]))
        furthest_distance_from_catcher = transformed_points[furthest_distance_from_catcher_index,2]
        z_points = np.linspace(0,furthest_distance_from_catcher,number_of_points_to_plot)
        x_trajectory,y_trajectory = self.__calculate_ball_trajectory_with_polynomial_fit(transformed_points)
        x_points = x_trajectory(z_points)
        y_points = y_trajectory(z_points)
        points = np.transpose(np.vstack((x_points,y_points,z_points)))
        return points
        
    def __calculate_camera_position_in_catcher_frame(self):
        homogenous_camera_position = np.array([0,0,0,1])
        transformed_camera_position = np.dot(self._camera_to_catcher_transformation, homogenous_camera_position)
        return transformed_camera_position
    
    def __calculate_ball_trajectory_with_polynomial_fit(self, points):
        """
        Calculates the ball trajectory using the given polynomial order.
        """
        x_polynomial_coeficients = np.polyfit(points[:,2] , points[:,0],self._degree)
        y_polynomial_coeficients = np.polyfit(points[:,2] , points[:,1],self._degree)
        x_trajectory = np.poly1d(x_polynomial_coeficients.flatten())
        y_trajectory = np.poly1d(y_polynomial_coeficients.flatten())
        return x_trajectory, y_trajectory

    def __transform_points_to_catcher_frame(self, points):
        """
        Transforms and returns the trajectory in the catcher frame.
        """
        number_of_points = np.shape(points)[0]
        homogenous_points = np.vstack((np.transpose(points), np.ones(number_of_points)))
        transformed_homogenous_points = np.dot(self._camera_to_catcher_transformation, homogenous_points)
        transformed_points = np.transpose(transformed_homogenous_points)[:,0:3]
        return transformed_points

    def __find_intersection_of_trajectory_and_catcher_plane(self, x_trajectory,y_trajectory):
        """
        Finds the x, y location of where the ball intersects the catcher
        """
        x_intersection_point = x_trajectory(0)
        y_intersection_point = y_trajectory(0)
        intersection_point = np.array([x_intersection_point,y_intersection_point])
        return intersection_point


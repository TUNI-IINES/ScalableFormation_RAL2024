#!/usr/bin/python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from math import pi
import numpy as np
from functools import partial
from tf_transformations import euler_from_quaternion
from nebolab_experiment_setup import NebolabSetup
from scenarios_unicycle.CCTA2024_Controller import SceneSetup



ROS_NODE_NAME = 'vicon_localization'

class ViconLoc(Node):

    def __init__(self):
        super().__init__(ROS_NODE_NAME)
        self.robot_count = SceneSetup.robot_num
        self.tb_l_si2uni = NebolabSetup.TB_L_SI2UNI
        self.pos_ahead_publishers = {}
        self.pos_center_publishers = {}
        self.pose_center = [Pose2D()]*self.robot_count
        self.pose_ahead = [Pose2D()]*self.robot_count


        for robot_index in range(self.robot_count):
            tb_name = f'tb3_{robot_index}'
            # Create pose publishers
            self.get_logger().info(f'Creating pos ahead publisher: /{tb_name}/pos')
            self.pos_ahead_publishers[tb_name] = \
                self.create_publisher(Pose2D, f'/{tb_name}/pos', 10)
            
            self.get_logger().info(f'Creating pos center publisher: /{tb_name}/posc')
            self.pos_center_publishers[tb_name] = \
                self.create_publisher(Pose2D, f'/{tb_name}/posc', 10)
            
            # Create pose subscriber (Vicon)
            self.get_logger().info(f'Creating Vicon subscriber: /vicon/{tb_name}/{tb_name}')

            self.pose_sub = self.create_subscription(
                Odometry,
                f'/{tb_name}/odom',
                partial(self.odom_callback, index=robot_index),
                10
            )
            self.pose_sub # Prevent unused variable warning
        

    def odom_callback(self, msg, index):
        q = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
             msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
                                            # quart[0] = roll
        quart = euler_from_quaternion(q)    # quart[1] = pitch
        p_angle = quart[2]               # quart[2] = yaw
        # make theta within from 0 to 360 degree
        if p_angle < 0:
            p_angle = p_angle + pi * 2
        if p_angle > pi * 2:
            p_angle = p_angle - pi * 2

        # COPIED AND ADAPTED FROM camerabased_localization.py --> localize_from_ceiling.compute_pose
        # Compute center point of wheel 
        self.pose_center[index].x = msg.pose.pose.position.x
        self.pose_center[index].y = msg.pose.pose.position.y
        self.pose_center[index].theta = p_angle

        # Compute point Ahead position
        self.pose_ahead[index].x = self.pose_center[index].x + self.tb_l_si2uni*np.cos(p_angle)
        self.pose_ahead[index].y = self.pose_center[index].y + self.tb_l_si2uni*np.sin(p_angle)
        self.pose_ahead[index].theta = p_angle
        
        self.pos_ahead_publishers[f"tb3_{index}"].publish(self.pose_ahead[index])
        self.pos_center_publishers[f"tb3_{index}"].publish(self.pose_center[index])


def main(args=None):
    rclpy.init(args=args)
    node = ViconLoc()
    try:
        node.get_logger().info("Spinning " + ROS_NODE_NAME)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

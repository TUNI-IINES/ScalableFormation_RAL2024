#!/usr/bin/python3
"""
spawn_turtlebot.py

Script used to spawn a turtlebot in a generic position
"""
import os
import sys
import rclpy
import yaml
import numpy as np
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity
from tf_transformations import quaternion_from_euler

SRC = "/home/kidovn/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/scripts/scenarios_unicycle/scenarios/"
# YAML_FILE = 'formation4_mixed'
# YAML_FILE = os.environ['YAML_NAME']
TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

def PyToGazConv(yaml_name):
    if not os.path.exists(f'{SRC}{yaml_name}.yml'):
            print(f"File {yaml_name} does not exist.")
            yaml_name = "formation4_mixed"

    with open(f'{SRC}{yaml_name}.yml', 'r') as file:
        scenario, _, _ = yaml.safe_load(file).values()

    form_l = scenario['formations']['form_scaling']
    form_size = [len(form) for form in scenario['formations']['links']]
    form_num = len(scenario['formations']['links'])
    form_id = np.array([idx for idx in range(form_num)
                        for _ in range(form_size[idx])])
    robot_num = len(scenario['formations']['structs'])
    struct = np.array([np.array(scenario['formations']['structs'][idx]) * form_l[form_id[idx]]
                        for idx in range(robot_num)])
    rot = lambda alpha: np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                                [np.sin(alpha), np.cos(alpha), 0.],
                                [0., 0., 1.]])

    init_pos = np.array([scenario['positions']['initial_positions'][form_id[idx]][:-1]+ rot(scenario['positions']
                                                                                            ['initial_positions'][form_id[idx]]
                                                                                            [-1]) @ struct[idx]
                        for idx in range(robot_num)
    ])

    init_pos_gaz ={f"tb3_{idx}": [*init_pos[idx], 0] for idx in range(robot_num)}

    return init_pos_gaz



class spawn_tb(Node):
    def __init__(self):
        super().__init__("entity_spawner")
        self.declare_parameter("yaml_name", "formation4_mixed")
        self.yaml_name = self.get_parameter("yaml_name").value

        self.get_logger().info(
            'Creating Service client to connect to `/spawn_entity`')
        client = self.create_client(SpawnEntity, "/spawn_entity")

        self.get_logger().info("Connecting to `/spawn_entity` service...")
        if not client.service_is_ready():
            client.wait_for_service()
            self.get_logger().info("...connected!")

        # Get path to the turtlebot3 burgerbot
        sdf_file_path = os.path.join(
            get_package_share_directory("turtlebot3_gazebo"), "models",
            "turtlebot3_"  + TURTLEBOT3_MODEL, "model.sdf")
        
        self.robot_dict = PyToGazConv(self.yaml_name)
        
        for robot in self.robot_dict:

            # Set data for request
            request = SpawnEntity.Request()
            request.name = robot

            request.xml = open(sdf_file_path, 'r').read()
            request.robot_namespace = robot
            request.initial_pose.position.x = float(self.robot_dict[robot][0])
            request.initial_pose.position.y = float(self.robot_dict[robot][1])
            request.initial_pose.position.z = 0.0
            quaternion = quaternion_from_euler(0.0, 0.0, float(self.robot_dict[robot][2]))
            request.initial_pose.orientation.x = quaternion[0]
            request.initial_pose.orientation.y = quaternion[1]
            request.initial_pose.orientation.z = quaternion[2]
            request.initial_pose.orientation.w = quaternion[3]

            self.get_logger().info("Sending service request to `/spawn_entity`")
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                print('response: %r' % future.result())
            else:
                raise RuntimeError(
                    'exception while calling service: %r' % future.exception())


def main(args=None):
    rclpy.init(args=args)

    node = spawn_tb()
    try:
        rclpy.spin(node) 
    except KeyboardInterrupt:
        pass
    node.get_logger().info("Done! Shutting down node.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
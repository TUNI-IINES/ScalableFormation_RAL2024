#!/usr/bin/python3
import os
import rclpy
from distutils.dir_util import copy_tree
import numpy as np
from rclpy.node import Node

SRC = f"/home/kidovn/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/scripts/scenarios_unicycle/scenarios/"
DST = f"/home/kidovn/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/"
WLD = f"/home/kidovn/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/empty_worlds"
# yaml_name = os.environ['YAML_NAME']

description = 'This SDF file is for enlarging environment'


class Wall:
    def __init__(self, vertex1, vertex2):
        self.__vertex1 = np.array(vertex1[:2])
        self.__vertex2 = np.array(vertex2[:2])

    @property
    def center(self):
        return (self.__vertex1 + self.__vertex2) / 2

    @property
    def length(self):
        return np.linalg.norm(self.__vertex2 - self.__vertex1)

    @property
    def angle(self):
        return np.arctan2(self.__vertex2[1] - self.__vertex1[1],
                          self.__vertex2[0] - self.__vertex1[0])


class converter(Node):
    def __init__(self, src, dst, wld):
        super().__init__("sdf_converter")
        self.declare_parameter("yaml_name", "formation4_mixed")
        self.yaml_name = self.get_parameter("yaml_name").value

        if not os.path.exists(f'{src}{self.yaml_name}.yml'):
            print(f"File {self.yaml_name} does not exist.")
            self.yaml_name = "formation4_mixed"

        self.get_logger().info(
            'Convert evironment obstacle file')
        with open(f'{src}{self.yaml_name}.yml', 'r') as file:
            import yaml
            scenario, _, _ = yaml.safe_load(file).values()

            # Set the static obstacles
            walls = [Wall(o[idx], o[idx + 1]) for o in scenario['obstacles'] for idx in range(len(o) - 1)]

        self.get_logger().info(
            'Make a new obstacle folder')

        # create a new file.txt to save the name of yaml_file
        file_name = "output.txt"

        full_path = os.path.join(src, file_name)
        with open(full_path, 'w') as file:
            file.write(self.yaml_name)

        # TODO: Copy folder/files
        copy_tree(f"{dst}turtlebot3_enlarge", f"{dst}{self.yaml_name}")

        with open(f'{dst}{self.yaml_name}/model.sdf', 'w') as file:
            obstacles = '\n'.join([f'''
            <collision name='obs_wall_{idx + 1}'>
                <pose>{w.center[0]} {w.center[1]} 0.25 0 0 {w.angle:.3f}</pose>
                <geometry>
                    <box><size>{w.length:.3f} 0.01 0.5</size></box>
                </geometry>
                <max_contacts>10</max_contacts>
                <surface><bounce/><friction><ode/></friction><contact><ode/></contact></surface>
            </collision>

            <visual name='obs_wall_{idx + 1}'>
                <pose>{w.center[0]} {w.center[1]} 0.25 0 0 {w.angle:.3f}</pose>
                <geometry>
                    <box><size>{w.length:.3f} 0.01 0.5</size></box>
                </geometry>
                <material>
                    <script>
                        <uri>file://media/materials/scripts/gazebo.material</uri>
                        <name>Gazebo/Bricks</name>
                    </script>
                </material>
            </visual>
            ''' for idx, w in enumerate(walls)])

            file.write(f'''
            <sdf version='1.5'>
                <!-- This SDF file is for enlarging environment -->
                <model name='ros_symbol'>
                    <static>1</static>
                    <link name='symbol'>
                        {obstacles}
                    </link>
                </model>
            </sdf>
            ''')

        self.get_logger().info(
            'Modify environment file')
        with open(f'{wld}/no_robot.model', 'w') as file:
            file.write(f'''
            <?xml version="1.0"?>
            <sdf version="1.6">
            <world name="default">

                <include>
                <uri>model://ground_plane</uri>
                </include>

                <include>
                <uri>model://sun</uri>
                </include>

                <scene>
                <shadows>false</shadows>
                </scene>

                <gui fullscreen='0'>
                <camera name='user_camera'>
                    <pose frame=''>0.319654 -0.235002 9.29441 0 1.5138 0.009599</pose>
                    <view_controller>orbit</view_controller>
                    <projection_type>perspective</projection_type>
                </camera>
                </gui>

                <physics type="ode">
                <real_time_update_rate>1000.0</real_time_update_rate>
                <max_step_size>0.001</max_step_size>
                <real_time_factor>1</real_time_factor>
                <ode>
                    <solver>
                    <type>quick</type>
                    <iters>150</iters>
                    <precon_iters>0</precon_iters>
                    <sor>1.400000</sor>
                    <use_dynamic_moi_rescaling>1</use_dynamic_moi_rescaling>
                    </solver>
                    <constraints>
                    <cfm>0.00001</cfm>
                    <erp>0.2</erp>
                    <contact_max_correcting_vel>2000.000000</contact_max_correcting_vel>
                    <contact_surface_layer>0.01000</contact_surface_layer>
                    </constraints>
                </ode>
                </physics>
                
                <model name="{self.yaml_name}">
                <static>1</static>
                <include>
                    <uri>model://{self.yaml_name}</uri>
                </include>
                </model>

            </world>
            </sdf>
            ''')


def main(args=None):
    rclpy.init(args=args)

    node = converter(src=SRC, dst=DST, wld=WLD)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.get_logger().info("Done! Shutting down node.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

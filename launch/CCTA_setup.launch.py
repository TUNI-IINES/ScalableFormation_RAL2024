#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Darby Lim

import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
   DeclareLaunchArgument('yaml_name', default_value='formation4_mixed', 
                         description='Name of the yaml file')

   
   converter = Node(
      package='turtlebot3_gazebo',
      executable='sdf_convert.py',
      name='sdf_converter',
      parameters=[{'yaml_name': LaunchConfiguration('yaml_name')}],
      )


   environment = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('turtlebot3_gazebo'), 'launch'),
         '/totally_empty.launch.py'])
      )

   spawn = Node(
      package='turtlebot3_gazebo',
      executable='spawn_turtlebot.py',
      name='entity_spawner',
      parameters=[{'yaml_name': LaunchConfiguration('yaml_name')}],
      )
   
   
   vicon = Node(
      package='turtlebot3_gazebo',
      executable='vicon_localization.py',
      name='vicon_localization'
      )
   # turtlebot3 = IncludeLaunchDescription(
   #    PythonLaunchDescriptionSource([os.path.join(
   #       get_package_share_directory('turtlebot3_gazebo'), 'launch'),
   #       '/spawn_tb3.launch.py'])
   #    )
   # vicon = IncludeLaunchDescription(
   #    PythonLaunchDescriptionSource([os.path.join(
   #       get_package_share_directory('turtlebot3_gazebo'), 'launch'),
   #       '/vicon_localization.launch.py'])
      # )
   # main = IncludeLaunchDescription(
   #    PythonLaunchDescriptionSource([os.path.join(
   #       get_package_share_directory('turtlebot3_gazebo'), 'launch'),
   #       '/expROS_main.launch.py'])
   #    )

   return LaunchDescription([
      converter,
      environment,
      spawn,
      vicon
   ])
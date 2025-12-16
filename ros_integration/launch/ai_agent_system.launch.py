"""
ROS2 Launch file for the complete AI Agent system with Gazebo.
Launches Gazebo simulation, AI agent node, and robot controller.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Package directory
    pkg_ai_agent_ros = get_package_share_directory('ai_agent_ros')
    
    # Launch arguments
    gui = LaunchConfiguration('gui', default='true')
    enable_vision = LaunchConfiguration('enable_vision', default='true')
    
    declare_gui_cmd = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Launch Gazebo GUI'
    )
    
    declare_enable_vision_cmd = DeclareLaunchArgument(
        'enable_vision',
        default_value='true',
        description='Enable vision module in AI agent'
    )
    
    # Include Gazebo simulation launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ai_agent_ros, 'launch', 'gazebo_simulation.launch.py')
        ),
        launch_arguments={'gui': gui}.items()
    )
    
    # AI Agent Node
    ai_agent_node = Node(
        package='ai_agent_ros',
        executable='ai_agent_node.py',
        name='ai_agent_node',
        output='screen',
        parameters=[{
            'enable_vision': enable_vision,
            'use_sim_time': True
        }]
    )
    
    # Robot Controller Node
    robot_controller_node = Node(
        package='ai_agent_ros',
        executable='robot_controller.py',
        name='robot_controller',
        output='screen',
        parameters=[{
            'use_sim_time': True
        }]
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_gui_cmd)
    ld.add_action(declare_enable_vision_cmd)
    
    # Add launch files and nodes
    ld.add_action(gazebo_launch)
    ld.add_action(ai_agent_node)
    ld.add_action(robot_controller_node)
    
    return ld

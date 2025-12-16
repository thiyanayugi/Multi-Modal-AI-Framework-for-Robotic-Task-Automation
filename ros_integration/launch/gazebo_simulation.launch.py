"""
ROS2 Launch file for Gazebo simulation with robot arm.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Package directories
    pkg_ai_agent_ros = get_package_share_directory('ai_agent_ros')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # Paths
    world_file = os.path.join(pkg_ai_agent_ros, 'worlds', 'blocks_world.world')
    urdf_file = os.path.join(pkg_ai_agent_ros, 'urdf', 'robot_arm.urdf.xacro')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    gui = LaunchConfiguration('gui', default='true')
    paused = LaunchConfiguration('paused', default='false')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_gui_cmd = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Set to "false" to run headless'
    )
    
    declare_paused_cmd = DeclareLaunchArgument(
        'paused',
        default_value='false',
        description='Start Gazebo paused'
    )
    
    # Gazebo server
    gzserver_cmd = ExecuteProcess(
        cmd=['gzserver',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             world_file],
        output='screen'
    )
    
    # Gazebo client
    gzclient_cmd = ExecuteProcess(
        condition=IfCondition(gui),
        cmd=['gzclient'],
        output='screen'
    )
    
    # Robot state publisher
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(urdf_file).read() if os.path.exists(urdf_file) else ''
        }],
        remappings=[('/joint_states', '/robot_arm/joint_states')]
    )
    
    # Spawn robot in Gazebo
    spawn_robot_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot_arm',
            '-file', urdf_file,
            '-x', '0',
            '-y', '0',
            '-z', '0'
        ],
        output='screen'
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_gui_cmd)
    ld.add_action(declare_paused_cmd)
    
    # Add nodes
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_robot_cmd)
    
    return ld

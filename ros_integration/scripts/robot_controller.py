#!/usr/bin/env python3
"""
Robot controller that executes actions from the AI Agent.
Controls the robot arm and gripper in Gazebo simulation (ROS2 version).
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Pose
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np


class RobotController(Node):
    """Controller for robot arm and gripper."""
    
    def __init__(self):
        """Initialize the robot controller."""
        super().__init__('robot_controller')
        self.get_logger().info("Initializing Robot Controller...")
        
        # Joint names
        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.gripper_joints = ['gripper_left_joint', 'gripper_right_joint']
        
        # Current state
        self.current_action = None
        self.is_executing = False
        
        # Publishers for joint position control
        self.joint1_pub = self.create_publisher(
            Float64,
            '/robot_arm/joint1_position_controller/command',
            10
        )
        self.joint2_pub = self.create_publisher(
            Float64,
            '/robot_arm/joint2_position_controller/command',
            10
        )
        self.joint3_pub = self.create_publisher(
            Float64,
            '/robot_arm/joint3_position_controller/command',
            10
        )
        
        # Gripper publishers
        self.gripper_left_pub = self.create_publisher(
            Float64,
            '/robot_arm/gripper_left_position_controller/command',
            10
        )
        self.gripper_right_pub = self.create_publisher(
            Float64,
            '/robot_arm/gripper_right_position_controller/command',
            10
        )
        
        # Status publisher
        self.status_pub = self.create_publisher(
            String,
            '/robot_controller/status',
            10
        )
        
        # Subscribers
        self.action_sub = self.create_subscription(
            String,
            '/ai_agent/action_plan',
            self.action_callback,
            10
        )
        
        # Predefined positions for blocks (in simulation)
        self.block_positions = {
            'red_block': {'joint1': 0.5, 'joint2': 0.3, 'joint3': -0.2},
            'blue_block': {'joint1': -0.5, 'joint2': 0.3, 'joint3': -0.2},
            'green_block': {'joint1': 0.5, 'joint2': -0.3, 'joint3': -0.2},
            'yellow_block': {'joint1': -0.5, 'joint2': -0.3, 'joint3': -0.2},
        }
        
        # Predefined destination positions
        self.destinations = {
            'left': {'joint1': -1.0, 'joint2': 0.0, 'joint3': 0.0},
            'right': {'joint1': 1.0, 'joint2': 0.0, 'joint3': 0.0},
            'center': {'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0},
            'corner': {'joint1': 0.8, 'joint2': 0.4, 'joint3': -0.3},
        }
        
        self.get_logger().info("✓ Robot Controller initialized")
        self.get_logger().info("Waiting for action commands...")
    
    def move_to_position(self, joint_positions, duration=2.0):
        """
        Move robot arm to specified joint positions.
        
        Args:
            joint_positions: Dict with joint names and target positions
            duration: Time to complete the movement (seconds)
        """
        self.get_logger().info(f"Moving to position: {joint_positions}")
        
        # Publish joint commands
        if 'joint1' in joint_positions:
            msg = Float64()
            msg.data = joint_positions['joint1']
            self.joint1_pub.publish(msg)
        if 'joint2' in joint_positions:
            msg = Float64()
            msg.data = joint_positions['joint2']
            self.joint2_pub.publish(msg)
        if 'joint3' in joint_positions:
            msg = Float64()
            msg.data = joint_positions['joint3']
            self.joint3_pub.publish(msg)
        
        # Wait for movement to complete
        import time
        time.sleep(duration)
    
    def open_gripper(self):
        """Open the gripper."""
        self.get_logger().info("Opening gripper")
        msg_left = Float64()
        msg_left.data = 0.03
        msg_right = Float64()
        msg_right.data = -0.03
        self.gripper_left_pub.publish(msg_left)
        self.gripper_right_pub.publish(msg_right)
        import time
        time.sleep(1.0)
    
    def close_gripper(self):
        """Close the gripper."""
        self.get_logger().info("Closing gripper")
        msg_left = Float64()
        msg_left.data = 0.0
        msg_right = Float64()
        msg_right.data = 0.0
        self.gripper_left_pub.publish(msg_left)
        self.gripper_right_pub.publish(msg_right)
        import time
        time.sleep(1.0)
    
    def pick_object(self, target_object):
        """
        Execute pick action for specified object.
        
        Args:
            target_object: Name of the object to pick
        """
        self.get_logger().info(f"Executing PICK action for: {target_object}")
        status_msg = String()
        status_msg.data = f"Picking {target_object}"
        self.status_pub.publish(status_msg)
        
        # Find object position
        object_key = None
        for key in self.block_positions.keys():
            if key.replace('_', ' ') in target_object.lower():
                object_key = key
                break
        
        if object_key is None:
            self.get_logger().warn(f"Unknown object: {target_object}, using default position")
            object_key = 'red_block'
        
        # Move to home position
        self.move_to_position({'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0}, duration=2.0)
        
        # Open gripper
        self.open_gripper()
        
        # Move to object
        self.move_to_position(self.block_positions[object_key], duration=3.0)
        
        # Close gripper
        self.close_gripper()
        
        # Lift object
        lift_pos = self.block_positions[object_key].copy()
        lift_pos['joint3'] = 0.2
        self.move_to_position(lift_pos, duration=2.0)
        
        self.get_logger().info(f"✓ Picked {target_object}")
    
    def place_object(self, destination):
        """
        Execute place action at specified destination.
        
        Args:
            destination: Name of the destination location
        """
        self.get_logger().info(f"Executing PLACE action at: {destination}")
        status_msg = String()
        status_msg.data = f"Placing at {destination}"
        self.status_pub.publish(status_msg)
        
        # Find destination position
        dest_key = None
        if destination:
            for key in self.destinations.keys():
                if key in destination.lower():
                    dest_key = key
                    break
        
        if dest_key is None:
            self.get_logger().warn(f"Unknown destination: {destination}, using center")
            dest_key = 'center'
        
        # Move to destination
        self.move_to_position(self.destinations[dest_key], duration=3.0)
        
        # Lower object
        place_pos = self.destinations[dest_key].copy()
        place_pos['joint3'] = -0.2
        self.move_to_position(place_pos, duration=2.0)
        
        # Open gripper
        self.open_gripper()
        
        # Move back to home
        self.move_to_position({'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0}, duration=2.0)
        
        self.get_logger().info(f"✓ Placed at {destination}")
    
    def sort_blocks(self):
        """Execute sorting action for all blocks."""
        self.get_logger().info("Executing SORT action")
        status_msg = String()
        status_msg.data = "Sorting blocks by color"
        self.status_pub.publish(status_msg)
        
        # Sort order: red, blue, green, yellow
        colors = ['red', 'blue', 'green', 'yellow']
        destinations_list = ['left', 'center', 'right', 'corner']
        
        for i, color in enumerate(colors):
            target = f"{color}_block"
            dest = destinations_list[i % len(destinations_list)]
            
            self.get_logger().info(f"Sorting {color} block to {dest}")
            self.pick_object(target)
            self.place_object(dest)
        
        self.get_logger().info("✓ Sorting complete")
    
    def action_callback(self, msg):
        """
        Callback for action plan messages.
        
        Args:
            msg: String message containing action plan
        """
        if self.is_executing:
            self.get_logger().warn("Already executing an action, ignoring new command")
            return
        
        self.is_executing = True
        action_plan = msg.data
        
        try:
            # Parse action from plan
            lines = action_plan.split('\n')
            action = None
            target = None
            destination = None
            
            for line in lines:
                if line.startswith('ACTION:'):
                    action = line.split(':', 1)[1].strip()
                elif line.startswith('TARGET:'):
                    target = line.split(':', 1)[1].strip()
                elif line.startswith('DESTINATION:'):
                    destination = line.split(':', 1)[1].strip()
            
            self.get_logger().info(f"\n{'='*60}")
            self.get_logger().info(f"Executing: {action} | Target: {target} | Dest: {destination}")
            self.get_logger().info(f"{'='*60}")
            
            # Execute action
            if action and 'pick' in action.lower():
                if target and target != 'None':
                    self.pick_object(target)
                    if destination and destination != 'None':
                        self.place_object(destination)
                else:
                    self.get_logger().warn("No target specified for pick action")
            
            elif action and 'place' in action.lower():
                if destination and destination != 'None':
                    self.place_object(destination)
                else:
                    self.get_logger().warn("No destination specified for place action")
            
            elif action and 'sort' in action.lower():
                self.sort_blocks()
            
            elif action and 'move' in action.lower():
                if target and target != 'None':
                    self.pick_object(target)
                    if destination and destination != 'None':
                        self.place_object(destination)
            
            else:
                self.get_logger().warn(f"Unknown action: {action}")
            
            status_msg = String()
            status_msg.data = "Action completed"
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error executing action: {e}")
            error_msg = String()
            error_msg.data = f"Error: {str(e)}"
            self.status_pub.publish(error_msg)
        
        finally:
            self.is_executing = False


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        controller = RobotController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
ROS2 node that bridges the AI Agent Framework with ROS2.
Receives natural language commands and publishes robot actions.
"""

import rclpy
from rclpy.node import Node
import sys
import os
from pathlib import Path
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion
from cv_bridge import CvBridge
import numpy as np
from PIL import Image as PILImage

# Add AI Agent Framework to path
# Get the actual path (resolve symlinks)
# This ensures the framework can be imported even when installed via symlink
script_path = Path(__file__).resolve()
# Go up from scripts/ -> ros_integration/ -> ai_agent_framework/
# Navigate directory structure to find the framework root
framework_path = script_path.parent.parent.parent
sys.path.insert(0, str(framework_path))

from src.agent import RoboticAgent
from src.vision_module import VisionModule


class AIAgentROSNode(Node):
    """ROS2 node that integrates AI Agent Framework with ROS2."""
    
    def __init__(self):
        """Initialize the ROS2 node and AI agent."""
        super().__init__('ai_agent_node')
        self.get_logger().info("Initializing AI Agent ROS2 Node...")
        
        # Initialize CV Bridge for image conversion
        # Converts between ROS2 Image messages and OpenCV/NumPy arrays
        self.bridge = CvBridge()
        
        # Store latest camera image
        # Cache the most recent image for use when processing commands
        self.latest_image = None
        
        # Initialize AI Agent
        # Load knowledge base from the framework directory
        knowledge_base_path = os.path.join(
            framework_path, 
            "knowledge_base/manipulation_strategies.json"
        )
        
        try:
            # Create RoboticAgent with vision enabled for multimodal understanding
            # This integrates language, vision, and RAG modules
            self.agent = RoboticAgent(
                knowledge_base_path=knowledge_base_path,
                enable_vision=True
            )
            self.get_logger().info("✓ AI Agent initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize AI Agent: {e}")
            raise
        
        # Publishers
        # Create ROS2 publishers for broadcasting agent outputs
        self.action_pub = self.create_publisher(
            String,
            '/ai_agent/action_plan',  # Publish generated action plans
            10
        )
        
        self.target_pose_pub = self.create_publisher(
            Pose,
            '/ai_agent/target_pose',  # Publish target poses for robot control
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/ai_agent/status',  # Publish processing status updates
            10
        )
        
        # Subscribers
        # Create ROS2 subscribers for receiving inputs
        self.command_sub = self.create_subscription(
            String,
            '/ai_agent/command',  # Subscribe to natural language commands
            self.command_callback,
            10
        )
        
        self.image_sub = self.create_subscription(
            Image,
            '/robot_arm/camera/image_raw',  # Subscribe to camera feed
            self.image_callback,
            10
        )
        
        self.get_logger().info("✓ ROS2 publishers and subscribers initialized")
        self.get_logger().info("=" * 60)
        self.get_logger().info("AI Agent ROS2 Node Ready!")
        self.get_logger().info("Send commands to: /ai_agent/command")
        self.get_logger().info("=" * 60)
    
    def image_callback(self, msg):
        """
        Callback for camera images.
        
        Args:
            msg: ROS2 Image message
        """
        try:
            # Convert ROS2 Image to OpenCV format
            # cv_bridge handles the conversion from ROS2 message to NumPy array
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            # Cache the latest image for use in command processing
            # This ensures we always use the most recent visual information
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def command_callback(self, msg):
        """
        Callback for natural language commands.
        
        Args:
            msg: ROS2 String message containing the command
        """
        command = msg.data
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"Received command: '{command}'")
        self.get_logger().info(f"{'='*60}")
        
        # Publish status
        # Inform subscribers that command processing has started
        status_msg = String()
        status_msg.data = "Processing command..."
        self.status_pub.publish(status_msg)
        
        try:
            # Process command with AI agent
            # Use visual context if available for multimodal understanding
            if self.latest_image is not None:
                self.get_logger().info("Using visual context from camera")
                result = self.agent.process_task(command, image=self.latest_image)
            else:
                self.get_logger().warn("No camera image available, processing without vision")
                result = self.agent.process_task(command)
            
            if result.success:
                self.get_logger().info("✓ Command processed successfully")
                
                # Publish action plan
                # Format result as structured message for robot controller
                action_plan_msg = String()
                action_plan_msg.data = f"ACTION: {result.parsed_command.action}\n"
                action_plan_msg.data += f"TARGET: {result.parsed_command.target_object}\n"
                action_plan_msg.data += f"DESTINATION: {result.parsed_command.destination}\n"
                action_plan_msg.data += f"CONFIDENCE: {result.parsed_command.confidence:.2f}\n\n"
                action_plan_msg.data += f"PLAN:\n{result.action_plan}"
                
                # Broadcast action plan to robot controller
                self.action_pub.publish(action_plan_msg)
                
                # Extract target position if available
                # Visual context may contain spatial information for navigation
                if result.visual_context:
                    self.get_logger().info(f"Visual context: {result.visual_context.get('workspace_description', 'N/A')}")
                
                # Publish success status
                # Notify subscribers that processing completed successfully
                success_msg = String()
                success_msg.data = "Command processed successfully"
                self.status_pub.publish(success_msg)
                
                # Log action plan
                self.get_logger().info("\n" + "="*60)
                self.get_logger().info("ACTION PLAN:")
                self.get_logger().info("="*60)
                self.get_logger().info(f"Action: {result.parsed_command.action}")
                self.get_logger().info(f"Target: {result.parsed_command.target_object}")
                self.get_logger().info(f"Destination: {result.parsed_command.destination}")
                self.get_logger().info(f"Confidence: {result.parsed_command.confidence:.2%}")
                self.get_logger().info("\nDetailed Plan:")
                for i, line in enumerate(result.action_plan.split('\n')[:10], 1):
                    if line.strip():
                        self.get_logger().info(f"  {line}")
                self.get_logger().info("="*60 + "\n")
                
            else:
                self.get_logger().error(f"Failed to process command: {result.error}")
                error_msg = String()
                error_msg.data = f"Error: {result.error}"
                self.status_pub.publish(error_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")
            error_msg = String()
            error_msg.data = f"Error: {str(e)}"
            self.status_pub.publish(error_msg)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        node = AIAgentROSNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

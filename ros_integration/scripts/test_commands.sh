#!/bin/bash
# Test script for sending commands to the AI Agent

echo "=========================================="
echo "AI Agent ROS - Test Commands"
echo "=========================================="
echo ""
echo "Make sure the system is running:"
echo "  roslaunch ai_agent_ros ai_agent_system.launch"
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "Test 1: Pick up red block"
echo "Command: 'Pick up the red block'"
rostopic pub -1 /ai_agent/command std_msgs/String "data: 'Pick up the red block'"
echo "Waiting 10 seconds..."
sleep 10

echo ""
echo "Test 2: Place in corner"
echo "Command: 'Place it in the corner'"
rostopic pub -1 /ai_agent/command std_msgs/String "data: 'Place it in the corner'"
echo "Waiting 10 seconds..."
sleep 10

echo ""
echo "Test 3: Move blue block"
echo "Command: 'Move the blue block to the left'"
rostopic pub -1 /ai_agent/command std_msgs/String "data: 'Move the blue block to the left'"
echo "Waiting 15 seconds..."
sleep 15

echo ""
echo "Test 4: Pick and place in one command"
echo "Command: 'Pick up the green block and place it in the center'"
rostopic pub -1 /ai_agent/command std_msgs/String "data: 'Pick up the green block and place it in the center'"
echo "Waiting 15 seconds..."
sleep 15

echo ""
echo "Test 5: Sort all blocks"
echo "Command: 'Sort all blocks by color'"
rostopic pub -1 /ai_agent/command std_msgs/String "data: 'Sort all blocks by color'"
echo "Waiting 60 seconds for sorting..."
sleep 60

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="


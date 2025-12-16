#!/bin/bash
# Simple script to send commands to the AI Agent

source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

if [ -z "$1" ]; then
    echo "Usage: $0 \"<command>\""
    echo ""
    echo "Examples:"
    echo "  $0 \"Pick up the red block\""
    echo "  $0 \"Move the blue block to the left\""
    echo "  $0 \"Sort all blocks by color\""
    exit 1
fi

COMMAND="$1"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   📡 Sending Command to AI Agent                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Command: $COMMAND"
echo ""
echo "Sending..."

rostopic pub -1 /ai_agent/command std_msgs/String "data: '$COMMAND'"

echo ""
echo "✅ Command sent!"
echo ""
echo "📊 Monitoring action plan (Ctrl+C to stop)..."
echo "════════════════════════════════════════════════════════════"
echo ""

rostopic echo /ai_agent/action_plan


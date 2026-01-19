# ROS2 Humble AI Agent Framework Dockerfile
# Multi-stage build for ROS2 Humble with AI/ML dependencies
FROM osrf/ros:humble-desktop-full

# Set environment variables
# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
# Ensure Python output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# Core development tools and ROS2 packages for robotic manipulation
RUN apt-get update && apt-get install -y \
    # Python and build tools
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    wget \
    curl \
    vim \
    nano \
    build-essential \
    cmake \
    # Gazebo dependencies for simulation
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    ros-${ROS_DISTRO}-gazebo-ros \
    # ROS2 control packages for robot manipulation
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-xacro \
    # Computer vision bridge for image processing
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    # X11 and display for Gazebo GUI
    x11-apps \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Create workspace
# Standard ROS2 workspace structure with src directory
RUN mkdir -p /root/ros2_ws/src
WORKDIR /root/ros2_ws

# Copy AI Agent Framework
# Copy entire project into the workspace
COPY . /root/ros2_ws/src/ai-framework/

# Install Python dependencies
# AI/ML packages for vision, language, and RAG modules
RUN pip3 install --no-cache-dir -r /root/ros2_ws/src/ai-framework/requirements.txt

# Create symbolic link for ROS package
# Link ros_integration directory to standard ROS package name
RUN ln -s /root/ros2_ws/src/ai-framework/ros_integration /root/ros2_ws/src/ai_agent_ros

# Initialize rosdep (if not already initialized)
# rosdep manages ROS package dependencies
RUN rosdep update || true

# Install ROS dependencies
# Resolve and install all ROS package dependencies
RUN cd /root/ros2_ws && \
    . /opt/ros/${ROS_DISTRO}/setup.sh && \
    rosdep install --from-paths src --ignore-src -r -y || true

# Build the workspace
# Compile all ROS2 packages with symlink-install for faster development
RUN cd /root/ros2_ws && \
    . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build --symlink-install

# Setup entrypoint
# Custom entrypoint script sources ROS2 environment before executing commands
COPY ros_integration/docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh || echo "#!/bin/bash\nset -e\nsource /opt/ros/${ROS_DISTRO}/setup.bash\nsource /root/ros2_ws/install/setup.bash\nexec \"$@\"" > /ros_entrypoint.sh && chmod +x /ros_entrypoint.sh

# Source ROS2 setup in bashrc
# Automatically source ROS2 environment in interactive shells
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Set entrypoint and default command
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]


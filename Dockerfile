# ROS2 Humble AI Agent Framework Dockerfile
FROM osrf/ros:humble-desktop-full

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
    # Gazebo dependencies
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    ros-${ROS_DISTRO}-gazebo-ros \
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    # X11 and display
    x11-apps \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir -p /root/ros2_ws/src
WORKDIR /root/ros2_ws

# Copy AI Agent Framework
COPY . /root/ros2_ws/src/robolingua/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /root/ros2_ws/src/robolingua/requirements.txt

# Create symbolic link for ROS package
RUN ln -s /root/ros2_ws/src/robolingua/ros_integration /root/ros2_ws/src/ai_agent_ros

# Initialize rosdep (if not already initialized)
RUN rosdep update || true

# Install ROS dependencies
RUN cd /root/ros2_ws && \
    . /opt/ros/${ROS_DISTRO}/setup.sh && \
    rosdep install --from-paths src --ignore-src -r -y || true

# Build the workspace
RUN cd /root/ros2_ws && \
    . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build --symlink-install

# Setup entrypoint
COPY ros_integration/docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh || echo "#!/bin/bash\nset -e\nsource /opt/ros/${ROS_DISTRO}/setup.bash\nsource /root/ros2_ws/install/setup.bash\nexec \"\$@\"" > /ros_entrypoint.sh && chmod +x /ros_entrypoint.sh

# Source ROS2 setup in bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

# ROS package for 3D object detection with tracking and trajectory prediction
This package allows users to use any 2D object detector and depth camera to predict 3D bounding boxes of detected objects, as well as track their movement and predict their trajectories.

This version supports the Melodic and Noetic distributions of ROS. The programs were tested using the Intel RealSense D435 camera and ROS Melodic and Noetic, although Noetic is recommended to avoid compatibility issues with different versions of Python. Detailed documentation can be found in the various Python scripts.

## Installation instructions
To use the package, the Intel RealSense SDK 2.0 and ROS Wrapper (https://github.com/IntelRealSense/realsense-ros) need to be installed.

In order to work, a 2D detector has to publish the 2D bounding box information to a specified topic.

## Usage instructions
To use the package, the user needs to specify the ROS topics of at least the depth image and 2D bounding boxes. This package is agnostic to the 2D detector used, as long as it publishes the 2D bounding box in the specified manner. Detailed instructions about the various topics are found in the main.py file.

For the Intel RealSense D435 camera, use "roslaunch realsense2_camera rs_aligned_depth.launch"
To launch the 3D bounding box estimation node, use "rosrun ros_3d_bb main.py".
To launch the tracking with prediction, use "rosrun ros_3d_bb tracker.py".

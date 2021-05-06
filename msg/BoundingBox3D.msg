# 3D bounding box message
# All units in meters
# This kind of message already exists in
# https://github.com/ros-perception/vision_msgs/blob/melodic-devel/msg/BoundingBox3D.msg
# but the corresponding array is not defined for ROS Melodic.

# Header to set a coordinate frame and timestamp
Header header

# 3D position and orientation
geometry_msgs/Pose center

# Size/shape of the bounding box
geometry_msgs/Vector3 size

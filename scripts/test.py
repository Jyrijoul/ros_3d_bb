#!/usr/bin/env python3

import rospy
# import roslib
import sys
import numpy as np
from sensor_msgs.msg import Image


VERBOSE = True


class ros_3d_bb:
    def __init__(self):
        image_topic = "/camera/color/image_raw"
        if VERBOSE:
            print("Initializing the 3D bounding-box module...")
            print("Trying to subscribe to topic " + image_topic + ".")

        self.image_sub = rospy.Subscriber(image_topic, Image,
                                          self.callback, queue_size=1)
        if VERBOSE:
            print("Subscribed to topic: " + image_topic)

    def callback(self, image_data):
        rospy.loginfo("Hello, ROS!")


def main(args):
    obj = ros_3d_bb()
    rospy.init_node("ros_3d_bb")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down the 3D bounding-box module...")


if __name__ == "__main__":
    main(sys.argv)

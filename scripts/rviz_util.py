#!/usr/bin/env python3

import rospy
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PointStamped, Pose, Point, Vector3, Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

# Some magic code to use Python3
#import sys
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import tf2_ros
from tf2_msgs.msg import TFMessage


class TFPublisher:
    def __init__(self, pose, parent_frame_id="base_link", child_frame_id="realsense_mount"):
        self.parent_frame_id = parent_frame_id
        self.child_frame_id = child_frame_id
        self.tf_pub = rospy.Publisher("/tf", TFMessage, queue_size=1)
        self.pose = pose

    def publish(self):
        t = TransformStamped()
        t.header.frame_id = self.parent_frame_id
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = self.child_frame_id

        t.transform.translation = self.pose.position
        t.transform.rotation = self.pose.orientation

        message = TFMessage([t])
        self.tf_pub.publish(message)


class RViz:
    def __init__(self, marker_topic="visualization_marker_array", frame_id="realsense_mount"):
        # Publishers
        self.marker_array_pub = rospy.Publisher(
            marker_topic, MarkerArray, queue_size=5)
        self.frame_id = frame_id
        self.marker_array = MarkerArray()
        self.trajectory_array = MarkerArray()
        self.marker_id = 0
        # Assume we have no more than 10000 / 4 = 2500 concurrent object detections.
        self.trajectory_id = 10000
        self.trajectories = {}

    # Source: https://docs.m2stud.io/cs/ros_additional/06-L3-rviz/
    def text(self, uid=0, x=0.0, y=0.0, z=1.1, text="", scaling=0.3, duration=1, alpha=0.9):
        if text == "":
            text = str(uid)
        marker = Marker(
            type=Marker.TEXT_VIEW_FACING,
            ns=str(uid),
            id=uid * 3,
            lifetime=rospy.Duration(duration),
            pose=Pose(Point(x, y, z), Quaternion(0, 0, 0, 1)),
            scale=Vector3(scaling, scaling, scaling),
            header=Header(frame_id=self.frame_id),
            color=ColorRGBA(0.0, 1.0, 0.0, alpha),
            text=text)
        self.marker_array.markers.append(marker)
        self.marker_id += 1

    def cylinder(self, uid=0, x=0.0, y=0.0, z=0, diameter=0.2, duration=1, alpha=0.9, trajectory=True):
        marker = Marker(
            type=Marker.CYLINDER,
            ns=str(uid),
            id=uid * 3 + 1,
            lifetime=rospy.Duration(duration),
            pose=Pose(Point(x, y, 0), Quaternion(0, 0, 0, 1)),
            scale=Vector3(diameter, diameter, z),
            header=Header(frame_id=self.frame_id),
            color=ColorRGBA(0.0, 0.0, 1.0, alpha))
        self.marker_array.markers.append(marker)
        self.marker_id += 1

        if trajectory:
            if not uid in self.trajectories:
                self.trajectories[uid] = Marker(
                type=Marker.LINE_STRIP,
                id=self.trajectory_id + uid,
                lifetime=rospy.Duration(duration),
                # pose=Pose(Point(x, y, z), Quaternion(0, 0, 0, 1)),
                points=[],
                header=Header(frame_id=self.frame_id),
                color=ColorRGBA(0.0, 1.0, 0.0, 0.3))

                self.trajectories[uid].scale.x = 0.01

            self.trajectories[uid].lifetime=rospy.Duration(duration)
            self.trajectories[uid].points.append(Point(x, y, 0))
            self.marker_array.markers.append(self.trajectories[uid])

    def arrow(self, uid=0, x=0.0, y=0.0, v_x=0.0, v_y=0.0, duration=1, alpha=0.9, r=1, g=0, b=0):
        marker = Marker(
            type=Marker.ARROW,
            ns=str(uid),
            id=uid * 3 + 2,
            lifetime=rospy.Duration(duration),
            points=[Point(x, y, 0), Point(v_x, v_y, 0)],
            # points=[Point(0, 0, 0), Point(2, 2, 0)],
            # pose=Pose(Point(x, y, 0), Quaternion(0, 0, 0, 1)),
            scale=Vector3(0.05, 0.1, 0),
            header=Header(frame_id=self.frame_id),
            color=ColorRGBA(r, g, b, alpha))
        self.marker_array.markers.append(marker)
        self.marker_id += 1

    def publish(self):
        # Publish the markers.
        self.marker_array_pub.publish(self.marker_array)
        # Empty the marker array and reset the id.
        # self.marker_array = MarkerArray()
        self.marker_array.markers.clear()
        self.marker_id = 0

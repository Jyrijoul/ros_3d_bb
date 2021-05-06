#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32MultiArray
import numpy as np
from scipy.spatial import distance
import time
import rviz_util
import predictor
from timer import Timer
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseStamped, Vector3Stamped
from ros_3d_bb.msg import BoundingBox3D, BoundingBox3DArray
import tf2_ros
import tf2_geometry_msgs


VERBOSE = True
DEBUG = False
TIME = True


class WorldPoint:
    def __init__(self, x=0, y=0, z=0, s_x=0, s_y=0, s_z=0, bounding_box=None):
        if not bounding_box:
            self.x = x
            self.y = y
            self.z = z
            self.s_x = s_x
            self.s_y = s_y
            self.s_z = s_z
        else:
            self.x = bounding_box.center.position.x
            self.y = bounding_box.center.position.y
            self.z = bounding_box.center.position.z
            self.s_x = bounding_box.size.x
            self.s_y = bounding_box.size.y
            self.s_z = bounding_box.size.z


class DetectedObject:
    def __init__(self, uid, world_point):
        # Set initial values in order to call self.update()
        self.x = world_point.x
        self.y = world_point.y
        
        self.update(world_point)

        self.uid = uid

        if DEBUG:
            print("Created new object:", self)

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def __str__(self):
        return "UID: " + str(self.uid) + ", x: " + str(self.x) + ", y: " + str(self.y) + ", v_x: " + str(self.v_x) + ", v_y: " + str(self.v_y)

    def update(self, world_point):
        """Updates the object's position, scale, calculates the velocity, resets disappeared counter."""
        x, y = world_point.x, world_point.y
        self.v_x = x - self.x
        self.v_y = y - self.y
        self.x = x
        self.y = y
        self.z = world_point.z
        self.diameter = world_point.s_x
        self.height = world_point.s_y
        self.disappeared = 0

    def has_disappeared(self):
        self.disappeared += 1


class Tracker:
    def __init__(self, max_frames_disappeared=30):
        self.max_frames_disappeared = max_frames_disappeared
        self.current_uid = 0

        self.objects = []

    def new_object(self, world_point):
        """Register a newly detected object."""
        self.objects.append(DetectedObject(self.current_uid, world_point))

        # Increment the UID.
        self.current_uid += 1

    def get_coordinates(self):
        coordinates = []
        for detected_object in self.objects:
            if DEBUG:
                print("Object:", detected_object)
            coordinates.append((detected_object.x, detected_object.y))

        return coordinates

    def delete_if_disappeared(self, detected_object):
        """Delete the long-disappeared objects (>= max_frames_disappeared)"""
        if detected_object.disappeared >= self.max_frames_disappeared:
            self.objects.remove(detected_object)

    def update(self, world_points: list):
        """Updates the positions of detected objects, add new objects and deletes old objects."""
        if len(world_points) == 0:
            for detected_object in self.objects:
                detected_object.disappeared()
                self.delete_if_disappeared(detected_object)
        elif len(self.objects) == 0:
            for world_point in world_points:
                self.new_object(world_point)
        else:
            current_coordinates = self.get_coordinates()
            distances = distance.cdist(
                np.array(current_coordinates), [(o.x, o.y) for o in world_points], "euclidean")

            # The following part is from the following article:
            # https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
            # Slightly modified the given code.
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            # Basically, this finds indices with least distances between the frames

            used_rows = set()
            used_cols = set()

            for i, j in zip(rows, cols):
                if i in used_rows or j in used_cols:
                    continue
                else:
                    self.objects[i].update(world_points[j])
                    used_rows.add(i)
                    used_cols.add(j)

            len_rows = distances.shape[0]
            len_cols = distances.shape[1]

            if len_rows > len_cols:
                unused_rows = list(set(
                    range(0, distances.shape[0])).difference(used_rows))
                # print("Unused rows:", unused_rows)
                for i in reversed(unused_rows):
                    obj = self.objects[i]
                    obj.has_disappeared()
                    self.delete_if_disappeared(obj)
            elif len_rows < len_cols:
                unused_cols = set(
                    range(0, distances.shape[1])).difference(used_cols)
                for j in unused_cols:
                    self.new_object(world_points[j])

        if VERBOSE:
            print("Objects:", list(map(str, self.objects)))

    def get_position_dict(self):
        position_dict = {}

        for detected_object in self.objects:
            position_dict[detected_object.uid] = (
                detected_object.x, detected_object.y)

        return position_dict

    def get_velocity_dict(self):
        velocity_dict = {}

        for detected_object in self.objects:
            velocity_dict[detected_object.uid] = (
                detected_object.v_x, detected_object.v_y)

        return velocity_dict


class RosTracker:
    def __init__(self):
        self.points = []
        self.max_frames_disappeared = 30
        self.tracker = Tracker(self.max_frames_disappeared)
        self.predictor = predictor.Predictor(self.tracker, 0.4)
        self.time = time.time()
        self.framerate = 0
        # For handling RViz visualization
        self.rviz = rviz_util.RViz(frame_id="world")

        self.frame_odom_id = "odom"
        pose_world = Pose()
        pose_world.orientation.w = 1
        self.tf_publisher_world = rviz_util.TFPublisher(
            pose_world, "world", "odom")
        self.frame_realsense_id = "realsense_mount"
        pose_realsense = Pose()
        # pose_realsense.orientation = Quaternion(0.5, -0.5, 0.5, -0.5)
        pose_realsense.orientation = Quaternion(0.5, -0.5, 0.5, -0.5)
        # pose_realsense.orientation.w = 1
        pose_realsense.position.x = 0.17
        pose_realsense.position.z = 0.20
        self.tf_publisher = rviz_util.TFPublisher(
            pose_realsense, "base_link", "realsense_mount")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        if TIME:
            self.timers = []

        # Subscribed topics
        self.bb_point_topic = "/ros_3d_bb/bb_3d"

        # Subscribers
        self.bb_point_sub = rospy.Subscriber(
            self.bb_point_topic, BoundingBox3DArray, self.bb_callback, queue_size=1
        )

        # Published topics
        self.points_topic = "/ros_3d_bb/"
        self.marker_topic = "visualization_marker"

        # Publishers
        # ---

        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + self.bb_point_topic)

    def update_framerate(self):
        current_time = time.time()
        self.framerate = 1 / (current_time - self.time)
        self.time = current_time

    def bb_callback(self, bb_array):
        if TIME:
            timer = Timer("update")
            self.timers.append(timer)

        self.tf_publisher.publish()
        self.tf_publisher_world.publish()

        trans = self.tf_buffer.lookup_transform(
            "world", self.frame_realsense_id, rospy.Time()
        )
        # print(trans)

        self.points = []
        for bounding_box in bb_array.boxes:
            # self.points.append(
            #     WorldPoint(bounding_box=bounding_box)
            # )
            pose_stamped = PoseStamped(
                bounding_box.header, bounding_box.center)
            vector3_stamped = Vector3Stamped(
                bounding_box.header, bounding_box.size)
            pose_transformed = tf2_geometry_msgs.do_transform_pose(
                pose_stamped, trans)
            # vector3_transformed = tf2_geometry_msgs.do_transform_vector3(
            #     vector3_stamped, trans)

            # new_bb = BoundingBox3D(
            #     bounding_box.header, pose_transformed.pose, vector3_transformed.vector)
            new_bb = BoundingBox3D(
                bounding_box.header, pose_transformed.pose, bounding_box.size)
            print(new_bb)
            self.points.append(
                WorldPoint(bounding_box=new_bb)
            )

        if TIME:
            timer.update()

        self.update_framerate()

        if TIME:
            timer.update()

        self.tracker.update(self.points)

        if TIME:
            timer.update()

        self.predictor.update()

        if TIME:
            timer.update()

        self.predictor.predict(self.framerate * 2)

        if TIME:
            timer.update()

        detections = self.tracker.objects

        # Visualization
        for obj in detections:
            # scaling = 1000  # From mm to m
            scaling = 1
            x = obj.x / scaling
            y = obj.y / scaling
            height = obj.height / scaling
            diameter = obj.diameter / scaling
            print("Diameter:", diameter)
            v_x = x + obj.v_x / scaling * self.framerate
            v_y = y + obj.v_y / scaling * self.framerate
            duration = self.max_frames_disappeared / self.framerate
            # Send data from the tracker
            self.rviz.text(obj.uid, x, y, duration=duration)
            self.rviz.cylinder(obj.uid, x, y, height, diameter,
                               duration=duration, alpha=0.5)
            self.rviz.arrow(obj.uid, x, y, v_x, v_y, duration=duration)
            # Send data from the predictor
            # Only the x and y
            predicted_x, predicted_y,  = self.predictor.predictions[obj.uid][:2]
            predicted_x /= scaling
            predicted_y /= scaling
            self.rviz.arrow(obj.uid + 1000, x, y, predicted_x,
                            predicted_y, duration=duration, r=0, g=1, b=0)

        self.rviz.publish()

        if TIME:
            timer.stop()

    def shutdown(self):
        if TIME:
            # for timer in self.timers:
            #     print(timer)
            averages = Timer.average_times(self.timers)
            rospy.loginfo("Average timings (in ms): " + str(averages))
        rospy.loginfo("Exiting...")


def main():
    rospy.init_node("ros_3d_bb_tracker")
    module = RosTracker()
    rospy.on_shutdown(module.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32MultiArray
import numpy as np
from scipy.spatial import distance
import time
import rviz_util
import predictor


VERBOSE = True
DEBUG = False


class CameraPoint:
    def __init__(self, point_from_camera):
        self.camera_x = point_from_camera[0]
        self.camera_y = point_from_camera[1]
        self.camera_z = point_from_camera[2]
        self.camera_s_x = point_from_camera[3]
        self.camera_s_y = point_from_camera[4]
        self.camera_s_z = point_from_camera[5]

    def to_bev_point(self):
        return BevPoint(self.camera_x, self.camera_z, self.camera_y,
                        self.camera_s_x, self.camera_s_z, self.camera_s_y)


class BevPoint:
    def __init__(self, x, y, z, s_x, s_y, s_z):
        self.x = x
        self.y = y
        self.z = z
        self.s_x = s_x
        self.s_y = s_y
        self.s_z = s_z

    def to_camera_point(self):
        return CameraPoint((self.x, self.z, self.y,
                            self.s_x, self.s_z, self.s_y))


class DetectedObject:
    def __init__(self, uid, bev_point=None, camera_point=None):
        if camera_point and not bev_point:
            bev_point = camera_point.to_bev_point()
        self.x = bev_point.x
        self.y = bev_point.y
        self.z = bev_point.z
        self.radius = bev_point.s_x
        self.height = bev_point.s_z

        self.v_x = 0
        self.v_y = 0
        self.uid = uid
        self.disappeared = 0

        if DEBUG:
            print("Created new object:", self)

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def __str__(self):
        return "UID: " + str(self.uid) + ", x: " + str(self.x) + ", y: " + str(self.y) + ", v_x: " + str(self.v_x) + ", v_y: " + str(self.v_y)

    def update(self, bev_point):
        """Updates the object's position, scale, calculates the velocity, resets disappeared counter."""
        x, y = bev_point.x, bev_point.y
        self.v_x = x - self.x
        self.v_y = y - self.y
        self.x = x
        self.y = y
        self.z = bev_point.z
        self.radius = bev_point.s_x
        self.height = bev_point.s_z
        self.disappeared = 0

    def has_disappeared(self):
        self.disappeared += 1


class Tracker:
    def __init__(self, max_frames_disappeared=30):
        self.max_frames_disappeared = max_frames_disappeared
        self.next_uid = 0

        self.objects = []

    def new_object(self, bev_point):
        """Register a newly detected object."""
        self.objects.append(DetectedObject(self.next_uid, bev_point))

        # Increment the UID.
        self.next_uid += 1

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

    def update(self, bev_points: list):
        """Updates the positions of detected objects, add new objects and deletes old objects."""
        if len(bev_points) == 0:
            for detected_object in self.objects:
                detected_object.disappeared()
                self.delete_if_disappeared(detected_object)
        elif len(self.objects) == 0:
            for bev_point in bev_points:
                self.new_object(bev_point)
        else:
            current_coordinates = self.get_coordinates()
            distances = distance.cdist(
                np.array(current_coordinates), [(o.x, o.y) for o in bev_points], "euclidean")

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
                    self.objects[i].update(bev_points[j])
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
                    self.new_object(bev_points[j])

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
        self.rviz = rviz_util.RViz()  # For handling RViz visualization

        # Subscribed topics
        self.bb_point_topic = "/ros_3d_bb/point"

        # Subscribers
        self.bb_point_sub = rospy.Subscriber(
            self.bb_point_topic, Int32MultiArray, self.bb_point_callback, queue_size=1
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

    def bb_point_callback(self, bb_point_multiarray):
        # bb_point_multiarray may contain zero to many points,
        # each described by 6 consecutive values.

        # Extract as many point x, y and z coordinates and corresponding scales as found:
        nr_of_bb_points = len(bb_point_multiarray.data) // 6
        self.points = []
        for i in range(nr_of_bb_points):
            self.points.append(
                CameraPoint((
                    bb_point_multiarray.data[0 + 6 * i],
                    bb_point_multiarray.data[1 + 6 * i],
                    bb_point_multiarray.data[2 + 6 * i],
                    bb_point_multiarray.data[3 + 6 * i],
                    bb_point_multiarray.data[4 + 6 * i],
                    bb_point_multiarray.data[5 + 6 * i],
                )).to_bev_point()
            )

        self.update_framerate()
        self.tracker.update(self.points)
        self.predictor.update()
        self.predictor.predict(self.framerate * 2)
        detections = self.tracker.objects

        # Visualization
        for obj in detections:
            scaling = 1000  # From mm to m
            x = obj.x / scaling
            y = obj.y / scaling
            height = obj.height / scaling
            # print("s_z:", z)
            radius = obj.radius / scaling
            v_x = x + obj.v_x / scaling * self.framerate
            v_y = y + obj.v_y / scaling * self.framerate
            duration = self.max_frames_disappeared / self.framerate
            # Data from the tracker
            self.rviz.text(obj.uid, x, y, duration=duration)
            self.rviz.cylinder(obj.uid, x, y, height, radius, duration=duration, alpha=0.5)
            self.rviz.arrow(obj.uid, x, y, v_x, v_y, duration=duration)
            # Data from the predictor
            predicted_x, predicted_y,  = self.predictor.predictions[obj.uid][:2]  # Only the x and y
            predicted_x /= scaling
            predicted_y /= scaling
            self.rviz.arrow(obj.uid + 1000, x, y, predicted_x, predicted_y, duration=duration, r=0, g=1, b=0)

        # if VERBOSE:
        #     rospy.loginfo(self.tracker.get_position_dict())

    def shutdown(self):
        print("Exiting...")


def main():
    rospy.init_node("ros_3d_bb_tracker")
    module = RosTracker()
    rospy.on_shutdown(module.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()

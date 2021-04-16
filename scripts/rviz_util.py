import rospy
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PointStamped, Pose, Point, Vector3, Quaternion
from visualization_msgs.msg import Marker


class RViz:
    def __init__(self, marker_topic="visualization_marker", max_concurrent_objects=1000):
        # Publishers
        self.marker_pub = rospy.Publisher(
            marker_topic, Marker, queue_size=5)
        self.max_concurrent_objects = max_concurrent_objects
    
    # Source: https://docs.m2stud.io/cs/ros_additional/06-L3-rviz/
    def text(self, uid=0, x=0.0, y=0.0, z=1.1, text="", scaling=0.3, duration=1, alpha=0.9):
        if text == "":
            text = str(uid)
        marker = Marker(
            type=Marker.TEXT_VIEW_FACING,
            id=uid,
            lifetime=rospy.Duration(duration),
            pose=Pose(Point(x, y, z), Quaternion(0, 0, 0, 1)),
            scale=Vector3(scaling, scaling, scaling),
            header=Header(frame_id='map'),
            color=ColorRGBA(0.0, 1.0, 0.0, alpha),
            text=text)
        self.marker_pub.publish(marker)

    def cylinder(self, uid=0, x=0.0, y=0.0, z=1, radius=0.5, duration=1, alpha=0.9):
        uid += self.max_concurrent_objects  # For not overwriting text and/or arrows
        marker = Marker(
            type=Marker.CYLINDER,
            id=uid,
            lifetime=rospy.Duration(duration),
            pose=Pose(Point(x, y, z / 2), Quaternion(0, 0, 0, 1)),
            scale=Vector3(radius, radius, z),
            header=Header(frame_id='map'),
            color=ColorRGBA(0.0, 0.0, 1.0, alpha))
        self.marker_pub.publish(marker)

    def arrow(self, uid=0, x=0.0, y=0.0, v_x=0.0, v_y=0.0, duration=1, alpha=0.9, r=1, g=0, b=0):
        uid += self.max_concurrent_objects * 2  # For not overwriting text and/or cylinders
        marker = Marker(
            type=Marker.ARROW,
            id=uid,
            lifetime=rospy.Duration(duration),
            points=[Point(x, y, 0), Point(v_x, v_y, 0)],
            # points=[Point(0, 0, 0), Point(2, 2, 0)],
            # pose=Pose(Point(x, y, 0), Quaternion(0, 0, 0, 1)),
            scale=Vector3(0.05, 0.1, 0),
            header=Header(frame_id='map'),
            color=ColorRGBA(r, g, b, alpha))
        self.marker_pub.publish(marker)

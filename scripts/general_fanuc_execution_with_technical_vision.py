#!/usr/bin/env python3

from __future__ import print_function
from six.moves import input
import numpy as np
import quaternion
import random
import sys
import matplotlib.pyplot as plt
import cv2
import pyrealsense2
import os
import time
from realsense_depth import *
import math
import statistics
import actionlib
import tf
import copy
import rospy
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Bool
import std_msgs
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

try:
    from math import pi, tau, dist, fabs, cos
except:  
    from math import pi, fabs, cos, sqrt
    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

def all_close(goal, actual, tolerance):
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False
    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)
    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        d = dist((x1, y1, z1), (x0, y0, z0))
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)
    return True

class MoveGroupPythonInterface(object):

    def __init__(self):
        super(MoveGroupPythonInterface, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface", anonymous=True)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=10,
        )
        planning_frame = move_group.get_planning_frame()
        eef_link = move_group.get_end_effector_link()
        group_names = robot.get_group_names()
        
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
    
    def go_to_pose_goal(self, x_length,y_length, z_length=0.2, q1=0.0, q2=3.14, q3=0.0):
        move_group = self.move_group
        quaternion = tf.transformations.quaternion_from_euler(q1, q2, q3)  # Поворот захвата
        pose_goal = geometry_msgs.msg.Pose()  
        pose_goal.orientation.x = quaternion[0]
        pose_goal.orientation.y = quaternion[1]
        pose_goal.orientation.z = quaternion[2]
        pose_goal.orientation.w = quaternion[3]
        pose_goal.position.x = x_length  
        pose_goal.position.y = y_length
        pose_goal.position.z = z_length
        move_group.set_pose_target(pose_goal)

        success = move_group.go(wait=True)
        move_group.stop()

        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def wait_for_state_update(
        self, box_is_known=False, box_is_attached=False, timeout=4
    ):
        box_name = self.box_name
        scene = self.scene
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0
            is_known = box_name in scene.get_known_object_names()
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True
            rospy.sleep(0.1)
            seconds = rospy.get_time()
        return False
    
    def add_box(self, name,x,y,z,a1=0.05,a2=0.05,a3=0.05,q3=0.0,timeout=4):
        box_name = self.box_name
        scene = self.scene
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link" 
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0,q3)
        box_pose.pose.orientation.x = quaternion[0]
        box_pose.pose.orientation.y = quaternion[1]
        box_pose.pose.orientation.z = quaternion[2]
        box_pose.pose.orientation.w = quaternion[3]
        box_name = str(name)
        box_pose.pose.position.x = x
        box_pose.pose.position.y = y
        box_pose.pose.position.z = z
        scene.add_box(box_name, box_pose, size=(a1, a2, a3))
        self.box_name = box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def attach_box(self, name,timeout=4):
        box_name = str(name)
        scene = self.scene
        eef_link = self.eef_link
        scene.attach_box(eef_link, box_name, touch_links=["tool0"])
        return self.wait_for_state_update(
            box_is_attached=True, box_is_known=False, timeout=timeout
        )

    def detach_box(self,name1, timeout=4):
        box_name = str(name1)
        scene = self.scene
        eef_link = self.eef_link
        scene.remove_attached_object(eef_link, name=box_name)
        return self.wait_for_state_update(
            box_is_known=True, box_is_attached=False, timeout=timeout
        )

    def remove_box(self, name, timeout=4):
        box_name = self.box_name
        scene = self.scene
        scene.remove_world_object(str(name))
        return self.wait_for_state_update(
            box_is_attached=False, box_is_known=False, timeout=timeout
        )

    def home(self):
        move_group = self.move_group
        move_group.set_named_target("all-zeros")
        plan_success, plan, planning_time, error_code = move_group.plan()
        move_group.execute(plan, wait=True)

def main():
    try:
        moveit = MoveGroupPythonInterface()
        
        moveit.add_box("wall_main_1",0.6,0.33,0.385,1.8,0.001,1.215)
        moveit.add_box("man_1",0.0,-0.5,0.45,0.2,0.2,0.9)
        moveit.add_box("man_2",0.29,-0.5,0.85,0.58,0.1,0.1)
        moveit.add_box("pod",0.0,-0.2,-0.1075,0.4,1.07,0.215)
        moveit.add_box("wall_main_2",-0.3,0,0.385,0.001,2,1.2)
        moveit.add_box("floor_main_2",0.0,0.0,-0.215,2,2,0.001)

        gripper_pub = rospy.Publisher('Gripper_control', Bool, queue_size=10, latch = True)
        gripper = Bool()

        dc = DepthCamera()

        object_parameters = {"Blue": [[0,0],[],[],[],0,[],0,[],[],0,0],
            "Red": [[0,0],[],[],[],0,[],0,[],[],0,0],
            "Yellow": [[0,0],[],[],[],0,[],0,[],[],0,0],
            "Orange": [[0,0],[],[],[],0,[],0,[],[],0,0],
            "Purple": [[0,0],[],[],[],0,[],0,[],[],0,0]
        }
        
        hsv_upper_and_lower = {
            "Blue": [[80, 100, 100], [100, 255, 255]],
            "Red": [[0, 120, 70], [10, 255, 255]],
            "Yellow": [[26, 50, 50], [35, 255, 255]],
            "Orange": [[11, 50, 50], [25, 255, 255]],
            "Purple": [[120, 50, 50], [170, 255, 255]]
        }

        offset_X = -0.245
        offset_Y = 0.32
        offset_angle = 20
        transition_coefficient = 0.001

        for i in range(40):

            contours = []
            ret, depth_frame, color_frame = dc.get_frame()
            img = color_frame
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv = cv2.blur(hsv, (5, 5))
            
            for key in object_parameters:

                lower = np.array(hsv_upper_and_lower[key][0])
                upper = np.array(hsv_upper_and_lower[key][1])
                mask = cv2.inRange(hsv, lower, upper)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=2)
                mask = cv2.dilate(mask, kernel, iterations=2)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours!=[] and contours is not None:
                    largest_contour = None
                    max_area = 0
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > max_area:
                            max_area = area
                            largest_contour = contour
                    
                    if cv2.contourArea(largest_contour) > 1000:
                        rect = cv2.minAreaRect(largest_contour)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        center_x = x + w // 2                          # центр объекта по X
                        center_y = y + h // 2                          # центр объекта по Y
                        angle = -rect[2]                               # угл поворота объекта
                        object_parameters[key][1].append(center_x)
                        object_parameters[key][2].append(center_y)
                        object_parameters[key][5].append(angle) 
                        
        for key in object_parameters:
            try:
                object_parameters[key][0][0] = statistics.mean(object_parameters[key][1])
                object_parameters[key][0][1] = statistics.mean(object_parameters[key][2])
                object_parameters[key][6] = statistics.mean(object_parameters[key][5])
                # Добавление объектов в сцену планирования
                moveit.add_box(key,(((object_parameters[key][0][1])*transition_coefficient)+offset_X),-(((1280-object_parameters[key][0][0])*transition_coefficient)+offset_Y),-0.19,0.05,0.05,0.05,q3 = ((object_parameters[key][6]+20)*3.14)/180)
                
            except:
                print('Параметры объекта цвета', key, 'не удалось получить')

        for key in object_parameters: # Перемещение объектов
            moveit.go_to_pose_goal((((object_parameters[key][0][1])*transition_coefficient)+offset_X),-(((1280-object_parameters[key][0][0])*transition_coefficient)+offset_Y),0.03, q3=((((object_parameters[key][6]+offset_angle)*3.14)/180)))
            moveit.attach_box(key)
            moveit.go_to_pose_goal((((object_parameters[key][0][1])*transition_coefficient)+offset_X),-(((1280-object_parameters[key][0][0])*transition_coefficient)+offset_Y),0.0, q3=((((object_parameters[key][6]+offset_angle)*3.14)/180)))
            gripper = True
            gripper_pub.publish(gripper)
            moveit.go_to_pose_goal(0.3,0.1,0.05, q3=((((object_parameters[key][6]+offset_angle)*3.14)/180)))
            moveit.detach_box(key)
            gripper = False
            gripper_pub.publish(gripper)
            moveit.home()

        for key in object_parameters:
            moveit.remove_box(key)

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    main()

from pyorca import ORCAAgent, orca
import time
import random
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
from controller import Supervisor
import torch
from torch import nn
from torchvision.utils import save_image
import sys
import os
import h5py
from datetime import datetime
from random import uniform
GOAL = None


class Orca_peds(object):

    def __init__(self, webot_ped, radius, max_speed=1.0):
        super(Orca_peds, self).__init__()
        self.ped = webot_ped
        self.position = np.asarray(webot_ped.getPosition())[[0,2]]
        self.velocity = np.asarray(webot_ped.getVelocity())[[0,2]]
        self.radius = radius
        self.max_speed = max_speed

    def update(self):
        self.position = np.asarray(self.ped.getPosition())[[0,2]]
        self.velocity = 0 #np.asarray(self.ped.getVelocity())[[0,2]]


class Orca (Supervisor):
    """Control a Hircus PROTO. with ORCA collision avoidance"""

    def __init__(self, radius=0.5, max_speed=0.7):
        """Constructor: initialize constants."""
        Supervisor.__init__(self)
        self.peds = []
        self.state = []
        self.max_speed = max_speed
        self.radius = radius
        self.goal = self.new_goal()

        i = 0
        while not self.getFromDef("Ped%d" % i) is None:
            self.peds.append(Orca_peds(self.getFromDef("Ped%d" % i), self.radius))
            i += 1
        self.logger = self.getFromDef("Logger")
        self.logger.getField('translation').setSFVec3f([self.goal[0], 10, self.goal[2]])
        self.hircus = self.getSelf()
        self.position = None
        self.velocity = None
        self.update()

        self.load_webots_devices()
        self.enable_sensors(DEVICE_SAMPLE_TIME)
        self.reset_motor_position()
        self.reset_motor_speed()

        self.stamp = 0.
        self.front_speed = 0.
        self.rear_speed = 0.
        self.left_speed = 0.
        self.right_speed = 0.
        self.front_steering_angle = 0.
        self.rear_steering_angle = 0.
        self.wheelbase = 0.38

        self.key = 0

    def load_webots_devices(self):
        self.left_motor = self.getDevice('left_wheel')
        self.right_motor = self.getDevice('right_wheel')
        self.front_motor = self.getDevice('front_wheel')
        self.rear_motor = self.getDevice('rear_wheel')
        self.front_steer = self.getDevice('front_steer')
        self.rear_steer = self.getDevice('rear_steer')
        self.gps = self.getDevice('gps')
        self.body_imu = self.getDevice('body_gyro')
        self.gnss_heading_device = self.getDevice('gnss_heading')
        self.camera = self.getDevice('CAM')
        self.Keyboard = self.getKeyboard()

    def enable_sensors(self, step_time):
        self.gps.enable(step_time)
        self.front_steer.getPositionSensor().enable(step_time)
        self.rear_steer.getPositionSensor().enable(step_time)
        self.body_imu.enable(step_time)
        self.gnss_heading_device.enable(step_time)
        self.Keyboard.enable(step_time)
        if not isinstance(self.camera, type(None)):
            self.camera.enable(step_time)
            self.height = self.camera.getHeight()
            self.width = self.camera.getWidth()
            self.fl = self.camera.getFocalLength()

    def reset_motor_position(self):
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.front_motor.setPosition(float('inf'))
        self.rear_motor.setPosition(float('inf'))
        self.front_steer.setPosition(float('inf'))
        self.rear_steer.setPosition(float('inf'))

    def reset_motor_speed(self):
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.front_motor.setVelocity(0)
        self.rear_motor.setVelocity(0)
        self.front_steer.setVelocity(0)
        self.rear_steer.setVelocity(0)

    def update_motors(self, speed, steer):
        if np.isnan(speed) or np.isnan(steer):
            # self.reset()
            steer = 0.
        self.left_motor.setVelocity(speed/WHEEL_RADIUS)
        self.right_motor.setVelocity(speed/WHEEL_RADIUS)
        self.front_motor.setVelocity(speed/WHEEL_RADIUS)
        self.rear_motor.setVelocity(speed/WHEEL_RADIUS)

        self.front_steer.setPosition(-steer)
        self.rear_steering_angle = steer
        self.rear_steer.setPosition(steer)
        self.front_steer.setVelocity(3)
        self.rear_steer.setVelocity(3)

    def checkreset(self):
        pos = self.hircus.getPosition()
        if np.sqrt(pos[0] ** 2 + pos[2] ** 2) >= 9.5:
            self.simulationReset()
            self.reset()
        # if self.getTime() > 50:
        #     self.reset()
        dist2goal = np.sqrt((pos[0] - GOAL[0]) ** 2 + (pos[2] - GOAL[2]) ** 2)
        if dist2goal < 0.5:
            # if within 0.5 [m] of goal
            print("Made it!!")
            self.reset()

    def reset(self):
        self.simulationReset()
        for ped in self.peds:
            ped.ped.restartController()
        self.logger.restartController()
        self.hircus.restartController()
        # new_goal()

    def new_goal(self):
        global GOAL
        GOAL = np.array([-3, 0, uniform(-3, 3)])
        print(GOAL)
        return GOAL

    def getPrefVelocity(self):
        heading = self.goal[[0,2]]-self.position
        mag = np.linalg.norm(heading)
        dir = heading/mag
        self.pref_velocity = dir * self.max_speed

    def update(self):
        self.get_position()
        self.get_velocity()
        self.getPrefVelocity()

    def get_position(self):
        self.position = np.asarray(self.hircus.getPosition())[[0,2]]

    def get_velocity(self):
        self.velocity = np.asarray(self.hircus.getVelocity())[[0,2]]

    def get_steer(self, new_vels):
        mag_velocity = np.linalg.norm(new_vels)
        unit_vector_1 = new_vels / mag_velocity
        yaw = self.gnss_heading_device.getRollPitchYaw()[2]
        unit_vector_2 = np.array([-np.cos(yaw), np.sin(yaw)])
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        angle_sin = - np.arcsin(np.cross(unit_vector_1, unit_vector_2))
        # print(angle, angle_sin)
        steer = np.arctan(self.wheelbase*angle_sin/mag_velocity)
        return steer

    def time_step(self):
        self.update()
        for ped in self.peds:
            ped.update()
        new_vels, _ = orca(self, self.peds, 1, 1/10)
        mag_new_vels = np.linalg.norm(new_vels)
        steer = self.get_steer(new_vels)
        self.update_motors(mag_new_vels, steer)
        self.checkreset()


if __name__ == "__main__":
    WHEEL_RADIUS = 0.16  # m
    WEBOTS_STEP_TIME = 100
    DEVICE_SAMPLE_TIME = int(WEBOTS_STEP_TIME)
    SCALE = 1000
    GNSS_RATE = 1
    WEBOTS_ROBOT_NAME = "CapraHircus"

    controller = Orca()
    # controller.simulationSetMode(0)
    while not controller.step(WEBOTS_STEP_TIME) == -1:
        controller.time_step()


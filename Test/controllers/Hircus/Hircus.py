import pandas as pd
from controller import Supervisor
import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
import sys
import os
import numpy as np
import pickle
# from transforms3d.euler import mat2euler, axangle2euler
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib as mpl
from random import uniform
from dataclasses import make_dataclass

dir_name = Path(Path.cwd()).parent.parent.parent
sys.path.insert(1, str(dir_name)+'/src')
from Badgrnet import HERDR
from actionplanner import HERDRPlan
from metrics_utils import plot_trajectory, plot_actions, plot_action_cam_view

WHEEL_RADIUS = 0.16  # m
WEBOTS_STEP_TIME = 100
DEVICE_SAMPLE_TIME = int(WEBOTS_STEP_TIME / 2) 
SCALE = 1000
GNSS_RATE = 1
HRZ = 20
BATCH = 50
GOAL = torch.tensor([uniform(-6, 3), uniform(-6, 6)]).repeat(BATCH, HRZ, 1)
# GOAL = torch.tensor[-0.6789, 3.25]).repeat(BATCH, HRZ, 1)
print(f"X: {GOAL[0,0,0]:.4f}, Z: {GOAL[0, 0, 1]:.4f}")
WEBOTS_ROBOT_NAME = "CapraHircus"
Ped_sample = make_dataclass("Sample", [("Actions", float), ("Ground_Truth", float), ("Image_Name", str)])
State_Event = make_dataclass("States", [("State", float), ("Event_Prob", float), ("Target_Pos", float)])


def new_goal():
    global GOAL
    GOAL = np.broadcast_to([uniform(-6, 6), uniform(-6, 6)], (BATCH, 2))
    print(GOAL[0])
    pass


class Hircus (Supervisor):
    """Control a Hircus PROTO."""

    def __init__(self, train=True):
        """Constructor: initialize constants."""
        Supervisor.__init__(self)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # self.model.classes = [0]
        self.df = pd.DataFrame(columns=["Actions", "Ground_Truth", "Image_Name"])
        self.stateR_df = pd.DataFrame(columns=["State", "Event_Prob", "Target_Pos"])
        self.train = train
        self.peds = []
        i = 0
        while not self.getFromDef("Ped%d" % i) is None:
            self.peds.append(self.getFromDef("Ped%d" % i))
            i += 1
        self.hircus = self.getSelf()
        self.pose = self.hircus.getPose()
        self.frame = None
        self.obj = []
        self.recog = []
        self.actions = []
        self.now = []
        self.event = []

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

        if not train:
            self.net = torch.load('Herdr_cross06-01-2022--18 50 17.pth', map_location=torch.device('cpu'))
            self.net.model_out = nn.Sequential(
                self.net.model_out,
                nn.Sigmoid()
            )
            self.net.eval()
        self.planner = HERDRPlan(Horizon=HRZ, vel_init=0.7)
        
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
            self.camera.recognitionEnable(step_time)
            self.height = self.camera.getHeight()
            self.width = self.camera.getWidth()
            
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
            self.reset()
        self.left_motor.setVelocity(speed/WHEEL_RADIUS)
        self.right_motor.setVelocity(speed/WHEEL_RADIUS)
        self.front_motor.setVelocity(speed/WHEEL_RADIUS)
        self.rear_motor.setVelocity(speed/WHEEL_RADIUS)

        self.front_steer.setPosition(-steer)
        self.rear_steering_angle = steer
        self.rear_steer.setPosition(steer)
        self.front_steer.setVelocity(1)
        self.rear_steer.setVelocity(1)
        
    def recognize(self):
        self.recog = self.camera.getRecognitionObjects()
        if self.recog:
            obj = self.camera.getRecognitionObjects()
            self.obj = [self.getFromId(node.get_id()) for node in obj]
            return 1
        else:
            return 0
    
    def reward(self):
        self.event = torch.zeros((BATCH, HRZ))
        robot_rot = self.gnss_heading_device.getRollPitchYaw()
        state = self.calculate_position(robot_rot)
        #  goalreward Shape: [BATCH, HRZ]
        goalReward = torch.sqrt(torch.square((state[:, :, 0]-GOAL[:, :, 0])) + torch.square((state[:, :, 2]-GOAL[:, :, 1])))
        if self.train and self.recognize():
            for ped in self.obj:
                ped_pos = torch.tensor(ped.getPosition())
                ped_pos = ped_pos.repeat(BATCH, HRZ, 1)
                SFRot = ped.getField("rotation").getSFRotation()
                ped_ori = torch.tensor(SFRot[0:3]) * SFRot[3]
                # Event Shape: [BATCH, HRZ]
                self.event = torch.logical_or(self.is_safe(state, ped_pos, ped_ori), self.event).float()
        elif not self.train:
            #  Event shape: [BATCH, HRZ]
            self.event = self.net(255*self.frame, self.actions)[:, :, 0].detach()
        plot_action_cam_view(self.actions, self.frame[0], self.event,
                             self.rear_steering_angle, self.front_motor.getVelocity()*WHEEL_RADIUS)
        cmap = mpl.cm.magma
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Probability')
        plt.show(block=False)
        plt.pause(0.05)
        # self.save_event(state)
        event_gain = goalReward.mean()*1.4
        reward = goalReward + event_gain * self.event
        return reward

    def save_event(self, state):
        toodf = pd.DataFrame([State_Event(state, self.event.detach().numpy(), GOAL[0, 0, :])])
        self.stateR_df = self.stateR_df.append(toodf, ignore_index=True)
        pass

    def calculate_position(self, rot):
        new_pos = torch.tensor(self.hircus.getPosition())
        new_state = torch.cat((new_pos.T, torch.tensor([-rot[2]])))
        batch_state = new_state.repeat(BATCH, HRZ, 1).transpose(1,2)
        # Y-axis is vertical, movement is in X-Z plane
        # [X Y Z Phi]
        for i in range(0, HRZ - 1):
            batch_state[:, 0, i + 1] = batch_state[:, 0, i] - (WEBOTS_STEP_TIME / SCALE) * torch.cos(
                batch_state[:, 3, i]) * self.actions[:, i, 0]
            batch_state[:, 2, i + 1] = batch_state[:, 2, i] - (WEBOTS_STEP_TIME / SCALE) * torch.sin(
                batch_state[:, 3, i]) * self.actions[:, i, 0]
            batch_state[:, 3, i + 1] = batch_state[:, 3, i] + (WEBOTS_STEP_TIME / SCALE) * self.actions[:, i, 1] * \
                                       self.actions[:, i, 0] / self.wheelbase
        ### output shape: [BATCH, HRZ, 4]
        return batch_state.permute(0, 2, 1)
        
    def is_safe(self, state, ped_pos, ped_ori):
        # Simple personal space model with an ellipse of radii "a" & "b" and offset by "shift"
        a = 0.8
        b = 1.5
        A = - ped_ori[1]
        shift = 1
        k = shift * torch.cos(A)
        h = - shift * torch.sin(A)
        first_term = torch.square(
            (state[:, :, 0] - ped_pos[:, :, 0] - h) * torch.cos(A) + (state[:, :, 2] - ped_pos[:, :, 2] - k) * torch.sin(
                A)) / a ** 2
        second_term = torch.square(
            (state[:, :, 0] - ped_pos[:, :, 0] - h) * torch.sin(A) - (state[:, :, 2] - ped_pos[:, :, 2] - k) * torch.cos(
                A)) / b ** 2
        check = (first_term + second_term) < 1
        return check.int()

    def reset(self):
        # self.exportImage(str(dir_name) + '/Test/controllers/Hircus/Topview.jpg', 100)
        # self.simulationReset()
        for ped in self.peds:
            ped.restartController()
        # add2pickle("Herdr_data_train.pkl", self.df)
        # add2pickle("State_rewards.pkl", self.stateR_df, overwrite=True)
        self.hircus.restartController()
        # new_goal()
        pass

    def checkreset(self):
        pos = self.hircus.getPosition()
        if np.sqrt(pos[0] ** 2 + pos[2] ** 2) >= 9.5:
            self.simulationReset()
            self.reset()
        # if self.getTime() > 50:
        #     self.reset()
        dist2goal = np.sqrt((pos[0] - GOAL[0, 0, 0]) ** 2 + (pos[2] - GOAL[0, 0, 1]) ** 2)
        if dist2goal < 0.50:
            # if within 0.5 [m] of goal
            print("Made it!!")
            self.reset()

    def todataset(self):
        # if self.recognize() and self.train:
        self.now = datetime.now()
        for i, sample in enumerate(self.actions):
            in2df = pd.DataFrame([Ped_sample(self.actions[i, :, :].detach(),
                                             self.event[i, :].detach(), f"{self.now}.png")])
            self.df = self.df.append(in2df, ignore_index=True)
        save_image(self.frame[0]/255, '%s.png' % ("./images/"+str(self.now)))

    def Herdr(self):
        frame = np.asarray(np.frombuffer(self.camera.getImage(), dtype=np.uint8))
        frame = np.reshape(np.ravel(frame), (self.height, self.width, 4), order='C')
        frame = torch.tensor(frame[:, :, 0:3]).float()
        self.frame = frame.permute(2, 0, 1).unsqueeze(0)
        self.frame = self.frame.repeat(BATCH, 1, 1, 1)
        self.actions = self.planner.sample_new(batches=BATCH)
        r = self.reward()
        if not self.train:
            best_r_arg = torch.argmin(torch.sum(r, dim=0))
        else:
            best_r_arg = torch.argmin(torch.sum(r, dim=0))
            # best_r_arg = torch.randint(0, BATCH, (1, 1)).item()

        # update motors and check for nan values
        self.update_motors(float(self.actions[best_r_arg, 0, 0]), float(self.actions[best_r_arg, 0, 1]))

        # Save To DataSet
        # self.todataset()

        # Update action mean and check for reset
        r = - r
        self.planner.update_new(r, self.actions)
        self.checkreset()

    @staticmethod
    def log(hircuspos, pedlist):
        dist_list = []
        collision = 0
        for person in pedlist:
            pos = person.getPosition()
            dist = np.sqrt((hircuspos[0]-pos[0])**2 + (hircuspos[2]-pos[2])**2)
            dist_list.append(dist)
            if dist < 0.5:
                collision = 1
        min_dist = np.asarray(dist_list).min()
        return np.asarray([min_dist, collision])


def add2pickle(file_name, dataframe, overwrite=False):
    if os.path.isfile(f'./{file_name}') and not overwrite:
        pkl_dataframe = pd.read_pickle(file_name)
        pkl_dataframe = pkl_dataframe.append(dataframe, ignore_index=True)
        pkl_dataframe.to_pickle(file_name, protocol=4)
    else:
        dataframe.to_pickle(file_name, protocol=4)
    pass


def pathlength(x, y):
    n = len(x)
    lv = [np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) for i in range(1, n)]
    L = sum(lv)
    return L


controller = Hircus(train=False)
# Hircus_traj = []
# ped_trajs = []
# min_dist = []
# in_collision = []
# unsafe = []  # "Score"
while not controller.step(WEBOTS_STEP_TIME) == -1:
    controller.Herdr()
    # Hircuspos = controller.hircus.getPosition()
    # Hircus_traj.append(Hircuspos)
    # if not controller.getTime() == 0.1:
    #     ped_trajs.append([p.getPosition() for p in controller.peds])
    # out_log = controller.log(Hircuspos, controller.peds)
    # min_dist.append(out_log[0])
    # in_collision.append(out_log[1])
    # unsafe.append(torch.sum(controller.event))
#
# unsafe = np.asarray(unsafe)
# min_dist = np.asarray(min_dist)
# Hircus_traj = np.asarray(Hircus_traj)
# ped_trajs = np.asarray(ped_trajs)
# traj_length = pathlength(Hircus_traj[:, 0], Hircus_traj[:, 2])

# plot_trajectory(Hircus_traj, min_dist, ped_trajs, GOAL, "Clearance", traj_length, collision=in_collision.count(1))
# plot_trajectory(Hircus_traj, unsafe, ped_trajs, GOAL, "Unsafe Score", traj_length, collision=in_collision.count(1))

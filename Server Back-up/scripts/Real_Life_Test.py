#!/usr/bin/env python
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode

from actionplanner import HERDRPlan
from Badgrnet import HERDR
# from torchvision import transforms
# from PIL import Image
from RL_config import get_params
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus
import rospy


def img_callback(img_msg, Control_Policy):
    bridge = CvBridge()
    img = torch.tensor(bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough'))
    # img = img.reshape((self.im_height, self.im_width, 4))
    img = img[:, :, :3]
    # '''Image shape [im_height, im_width, 3], Torch tensor BGR'''
    Control_Policy.frame = img.permute(2, 0, 1).float()
    # rospy.loginfo(img.shape)
    Control_Policy.Step()

def speed_callback(status, Control_Policy):
    Control_Policy.speed = status.velocity
    # rospy.loginfo(status.header)


class HerdrAgent(CompatibleNode):
    def __init__(self):
        super(HerdrAgent, self).__init__("Herdr Planner")

    def initialize(self, params):
        self.params = params
        # self.params = get_params()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.planner = HERDRPlan(self.params["Horizon"], self.params['Initial Speed'], self.params['Initial Steer Angle'],
            self.params['Gamma'], self.params['Action Sample Var'])
        self.Get_Model()
        self.Get_Goal()
        # rospy.init_node('Herdr', anonymous=True)
        self.control_pub = rospy.Publisher('/carla/ego/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=10)
        self.done = False
        self.speed = 0.
        
    def reset(self):
        self.planner.reset()

    def Get_Model(self):
        dir_name = str(Path(Path.cwd()))
        model_path = f"/home/nathan/catkin_ws/src/Herdr_test/models/{self.params['Model Name']}"
        self.model = torch.load(model_path, map_location=self.device)
        self.model.model_out = nn.Sequential(
                    self.model.model_out,
                    nn.Sigmoid())
        self.model.eval()
        self.model.to(self.device)

    def Get_Image(self):
        ''' Retrive Image from ROS '''
        image = rospy.wait_for_message('/carla/ego/front/image', Image)
        bridge = CvBridge()
        img = torch.tensor(bridge.imgmsg_to_cv2(image, desired_encoding='passthrough'))
        rospy.loginfo(img.shape)
        # rospy.loginfo(img.shape)
        # img = img.reshape((self.im_height, self.im_width, 4))
        img = img[:, :, :3]
        # '''Image shape [im_height, im_width, 3], Torch tensor BGR'''
        self.frame = img.permute(2, 0, 1).float()

    def Get_Interupt(self):
        ''' Get Stop Signal from ROS if present '''
        ## TODO Implement (might not need)
        pass

    def Get_Position(self):
        ''' Retrive Position and Orientation from ROS '''
        self.Position = torch.zeros((2))
        self.Rotation = torch.zeros((3))
        ## TODO Implement
        pass

    def Get_Goal(self):
        self.goal = torch.zeros((2)).repeat(self.params['Batches'],1,1)
        ## TODO Implement
        pass

    def Set_Actions(self):
        ''' Send actions over ROS '''
        if self.planner.mean[0, 0] > self.speed:
            a = 0.8
        else:
            a = 0.
        controls = CarlaEgoVehicleControl()
        controls.throttle = a
        controls.steer = -self.planner.mean[1,0].item()
        self.control_pub.publish(controls)

    def Propogate_States(self):
        new_pos = self.Position
        rotation = self.Rotation
        new_state = torch.cat((new_pos, rotation))
        batch_state = new_state.repeat(self.params['Batches'], self.params['Horizon'], 1).transpose(1, 2)
        dt = 1/self.params['Control Freq']
        ''' [X Y Z Phi] '''
        for i in range(0, self.params['Horizon'] - 1):
            batch_state[:, 0, i + 1] = batch_state[:, 0, i] + dt * torch.cos(
                batch_state[:, 3, i]) * self.actions[:, i, 0]
            batch_state[:, 1, i + 1] = batch_state[:, 1, i] + dt * torch.sin(
                batch_state[:, 3, i]) * self.actions[:, i, 0]
            batch_state[:, 3, i + 1] = batch_state[:, 3, i] - dt * self.actions[:, i, 1] * \
                                       self.actions[:, i, 0] / self.params['Wheel Base']
        
        ## TODO Check rotation frame of robot and verify state is updating correctly
        
        ''' Output shape: [BATCH, HRZ, 4] '''
        return batch_state.permute(0, 2, 1)

    def Call_Model(self):
        img = self.frame.unsqueeze(0)
        event = self.model(img.to(self.device), self.actions.to(self.device))[:, :, 0].detach().cpu()

        # self.state = self.Propogate_States()


        ''' goal_cost Shape: [BATCH, HRZ] '''
        # goal_cost = torch.linalg.norm(self.state[:,:,:2]-self.goal, dim=2)
        # goal_cost = (goal_cost - goal_cost.min()) / (goal_cost.max() - goal_cost.min())
        
        action_cost = self.actions[:,:,1] ** 2 / 2  + (self.actions[:,:,0] - 1.0) ** 2 / 2
        # self.score = self.params['Goal Cost Gain'] * goal_cost + event + self.params['Action Cost Gain'] * action_cost
        self.score = event + self.params['Action Cost Gain'] * action_cost
        return - self.score

    def Finish_Check(self):
        ''' Check if Robot is within distance to goal location '''
        
        dist2goal = torch.linalg.norm(self.Position - self.goal[0])
        if dist2goal <= 1.5:
            self.done = True
            self.success = 1.
            print('Made it!!!')

    def Step(self):
        # self.Get_Interupt()
        # self.Get_Image()
        # self.Get_Position()

        self.actions = self.planner.sample_new(batches=self.params['Batches'])
        score = self.Call_Model()
        self.planner.update_new(score, self.actions)
        
        self.Set_Actions()

        # self.Finish_Check()

    

if __name__ == '__main__':
    planner = HerdrAgent()
    parameters = {}
    executor = roscomp.executors.MultiThreadedExecutor()
    executor.add_node(planner)

    parameters['Control Freq'] = planner.get_param('control_freq', 5)
    parameters['Batches'] = planner.get_param('batches', 50)
    parameters['Horizon'] = planner.get_param('horizon', 10)
    parameters['Initial Speed'] = planner.get_param('initial_velocity', 1.5)
    parameters['Initial Steer Angle'] = planner.get_param('initial_steer', 0.0)
    parameters['Gamma' ]= planner.get_param('gamma', 20)
    parameters['Action Sample Var'] = planner.get_param('action_variance', (0.3,1.5))
    parameters['Goal Cost Gain'] = planner.get_param('goal_gain', 0.25)
    parameters['Action Cost Gain' ]= planner.get_param('action_gain', 0.2)
    parameters['Wheel Base'] = planner.get_param('wheel_base', 0.7)
    parameters['Model Name'] = planner.get_param('model_name', 'carla23-04-2022--14:57--from09:34.pth')

    planner.initialize(parameters)
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('input_image', Image, img_callback, planner)
    rospy.Subscriber('input_vel', CarlaEgoVehicleStatus, speed_callback, planner)
    # s = rospy.Service('image_server', Get_Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    # while not agent.done:
    #     agent.Step()

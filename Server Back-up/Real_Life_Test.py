
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from actionplanner import HERDRPlan
from Badgrnet import HERDR
# from torchvision import transforms
# from PIL import Image
from RL_config import get_params


class HerdrAgent():
    def __init__(self):
        self.params = get_params()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.planner = HERDRPlan(self.params["Horizon"], self.params['Initial Speed'], self.params['Initial Steer Angle'],
            self.params['Gamma'], self.params['Action Sample Var'])
        self.Get_Model()
        self.Get_Goal()
        
    def reset(self):
        self.planner.reset()

    def Get_Model(self):
        dir_name = str(Path(Path.cwd()))
        model_path = dir_name + f"/models/{self.params['Model Name']}"
        self.model = torch.load(model_path, map_location=self.device)
        self.model.model_out = nn.Sequential(
                    self.model.model_out,
                    nn.Sigmoid())
        self.model.eval()
        self.model.to(self.device)

    def Get_Image(self):
        ''' Retrive Image from ROS '''
        ## TODO Implement
        self.frame = torch.zeros(3,480,640)
        pass

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
        ## TODO Implement
        pass

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

        self.state = self.Propogate_States()


        ''' goal_cost Shape: [BATCH, HRZ] '''
        goal_cost = torch.linalg.norm(self.state[:,:,:2]-self.goal, dim=2)
        goal_cost = (goal_cost - goal_cost.min()) / (goal_cost.max() - goal_cost.min())
        
        action_cost = self.actions[:,:,1] ** 2 / 2  + (self.actions[:,:,0] - 1.0) ** 2 / 2
        self.score = self.params['Goal Cost Gain'] * goal_cost + event + self.params['Action Cost Gain'] * action_cost

        return - self.score

    def Finish_Check(self):
        ''' Check if Robot is within distance to goal location '''
        
        dist2goal = torch.linalg.norm(self.Position - self.goal[0])
        if dist2goal <= 1.5:
            self.done = True
            self.success = 1.
            print('Made it!!!')

    def Step(self):
        self.Get_Interupt()
        self.Get_Image()
        self.Get_Position()

        self.actions = self.planner.sample_new(batches=self.params['Batches'])
        score = self.Call_Model()
        self.planner.update_new(score, self.actions)
        
        self.Set_Actions()

        self.Finish_Check()

    

if __name__ == '__main__':
    agent = HerdrAgent()
    agent.Step()

import pandas as pd
import torch
from torch import nn
import torchvision
import cv2
import numpy as np
from pathlib import Path
import os
import pickle
from matplotlib import pyplot as plt
from matplotlib import figure
import matplotlib as mpl
from matplotlib.collections import LineCollection
from Badgrnet import HERDR
from actionplanner import HERDRPlan
from Carla_Trainer import carla_hdf5dataclass
from torch.utils.data.sampler import SubsetRandomSampler
import time


def plot_trajectory(robot_traj, line_values, goal, traj_length=-1, collision=-1):
    plt.clf()
    points = np.array([robot_traj[:, 1], robot_traj[:, 0]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('YlOrRd'), norm=plt.Normalize(0, line_values.max()))
    lc.set_array(line_values)
    lc.set_linewidth(3)
    plt.figure(1)
    plt.gca().add_collection(lc)
    # img = plt.imread("./Topview.jpg")  #/Users/NathanDurocher/Documents/GitHub/HERDR/
    # plt.imshow(img, extent=[-10, 10, -10, 10])
    # plt.autoscale(False)
    # plt.scatter(goal[0, 0, 1], goal[0, 0, 0], s=200, c='green', marker="o")
    plt.scatter(goal[1], goal[0], s=200, c='green', marker="o")
    # try:
    #     for i, ped in enumerate(ped_traj):
    #         if i == 0:
    #             plt.scatter(ped[:, 2], ped[:, 0], s=1., c='cyan', marker="o")
    #         else:
    #             plt.scatter(ped[:, 2], ped[:, 0], s=1., c='blue', marker="o")
    # except:
    #     pass
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    plt.axis('equal')
    # plt.colorbar(mappable=lc, label='%s from Peds (m)' % label)
    plt.xlabel('Y-Position (m)')
    plt.ylabel('X-Position (m)')
    plt.title('A Trajectory')
    if collision != -1:
        plt.figtext(0.03, 0.02, "# of Collisions: %d" % collision, c='red')
    if traj_length != -1:
        plt.figtext(0.70, 0.02, "Distance Travelled: %2.2f" % traj_length, c='red')


def plot_actions(position, line_values, location, GOAL, frame=None):
    plt.cla()
    points = np.expand_dims(np.array([position[:, :, 1].numpy(), position[:, :, 0].numpy()]).T, 2).transpose(1,0,2,3)
    segments = np.concatenate([points[:, :-1], points[:, 1:]], axis=2).reshape([-1,2,2])
    line_values = line_values[:,:-1].flatten().numpy()
    lc = LineCollection(segments, cmap=plt.get_cmap('YlOrRd'), norm=plt.Normalize(0, 1))
    lc.set_array(line_values.T)
    lc.set_linewidth(1)
    plt.gca().add_collection(lc)
    indices = torch.tensor([2, 1, 0])
    frame = torch.index_select(frame, 2, indices)
    x_min, x_max = location[0] - location[2], location[0] + location[2]
    y_min, y_max = location[1] - location[2], location[1] + location[2]
    plt.imshow(frame.int().numpy(), extent=[y_min, y_max, x_min, x_max])
    plt.autoscale(False)
    plt.xlabel('Y-Position (m)')
    plt.ylabel('X-Position (m)')
    plt.title(f'Top view: Its working')
    plt.scatter(GOAL[1], GOAL[0], s=20, c='red', marker="o")


def plot_action_cam_view(position, frame, event_probs, state):
    plt.cla()
    ''' state = [batch,horizon,(x,y,z,phi)]
        Transform from global to local co-ords'''
    omega = state[0, 0, 3] 
    transform = torch.tensor([[torch.cos(omega), -torch.sin(omega)],[torch.sin(omega), torch.cos(omega)]])
    state1 = torch.stack((state[:, :, 0],state[:, :, 1]),dim=2) - position[:2]
    state1 = torch.matmul(state1, transform.unsqueeze(0))

    '''rescale x,y axes to look more 3D'''
    state1[:, :, 1] = state1[:, :, 1]/abs(state1[:, :, 1]).max()*1.2
    state1[:, :, 0] = state1[:, :, 0]/abs(state1[:, :, 0]).max()*0.8

    ''' Plot multiple coloured line collections of safety score at calculated poisition '''
    points = np.expand_dims(np.array([state1[:, :, 1].numpy(), state1[:, :, 0].numpy()]).T, 2).transpose(1,0,2,3)
    segments = np.concatenate([points[:, :-1], points[:, 1:]], axis=2).reshape([-1,2,2])
    event_probs = event_probs[:,:-1].flatten().numpy()
    lc = LineCollection(segments, cmap=plt.get_cmap('YlOrRd'), norm=plt.Normalize(0, 1))
    lc.set_array(event_probs.T)
    lc.set_linewidth(1)
    plt.gca().add_collection(lc)
    indices = torch.tensor([2, 1, 0])
    frame = torch.index_select(frame, 2, indices)    
    plt.imshow(frame.int().numpy(), extent=[-1.5, 1.5, 0, 2])
    plt.autoscale(False)
    plt.title('Probabilities of Unsafe Position')


def count_data_ratio(loader):
    total = len(loader)*10
    positive = sum([torch.count_nonzero(gnd) for im, act, gnd in loader])
    ratio = total/positive
    print(f'Total Samples: {total}, # Positive: {positive} Ratio of total:positive {ratio:.4f}')

    


if __name__ == "__main__":
    dir_name = Path(Path.cwd())
    dir_name = str(dir_name) + '/carla_hdf5s/'
    dataset = carla_hdf5dataclass(dir_name, 10, load_all_files=True)
    print(len(dataset))
    test_sampler = SubsetRandomSampler(dataset.valid_start_indices)
    testloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, batch_size=1)
    count_data_ratio(testloader)
    # for index, row in df.iterrows():
    #     plot_actions(row['State'], row['Event_Prob'], "50", row['Target_Pos'], dir_name)

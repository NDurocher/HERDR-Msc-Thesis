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


def plot_trajectory(robot_traj, line_values, ped_traj, goal, label, traj_length, collision=-1):
    plt.clf()
    points = np.array([robot_traj[:, 2], robot_traj[:, 0]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('YlOrRd'), norm=plt.Normalize(0, line_values.max()))
    lc.set_array(line_values)
    lc.set_linewidth(3)
    plt.figure(1)
    plt.gca().add_collection(lc)
    img = plt.imread("./Topview.jpg")  #/Users/NathanDurocher/Documents/GitHub/HERDR/
    plt.imshow(img, extent=[-10, 10, -10, 10])
    plt.autoscale(False)
    # plt.scatter(goal[0, 0, 1], goal[0, 0, 0], s=200, c='green', marker="o")
    plt.scatter(goal[2], goal[0], s=200, c='green', marker="o")
    try:
        for i, ped in enumerate(ped_traj):
            if i == 0:
                plt.scatter(ped[:, 2], ped[:, 0], s=1., c='cyan', marker="o")
            else:
                plt.scatter(ped[:, 2], ped[:, 0], s=1., c='blue', marker="o")
    except:
        pass
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    # plt.axis('equal')
    plt.colorbar(mappable=lc, label='%s from Peds (m)' % label)
    plt.xlabel('Z-Position (m)')
    plt.ylabel('X-Position (m)')
    plt.title('%s from Pedestrians along a Trajectory' % label)
    if collision != -1:
        plt.figtext(0.03, 0.02, "# of Collisions: %d" % collision, c='red')
    plt.figtext(0.70, 0.02, "Distance Travelled: %2.2f" % traj_length, c='red')
    # plt.show()

def plot_actions(position, line_values, label, GOAL, directory_name):
    plt.clf()
    for pos, lv in zip(position.transpose(1, 0, 2), line_values):
        points = np.expand_dims(np.array([pos[:, 2], pos[:, 0]]).T, 1)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap('YlOrRd'), norm=plt.Normalize(0, 1))
        lc.set_array(lv.T)
        lc.set_linewidth(1)
        plt.gca().add_collection(lc)
    # plt.axis('equal')
    img = plt.imread(f"{directory_name}Topview.jpg")  # /Users/NathanDurocher/Documents/GitHub/HERDR/
    plt.imshow(img, extent=[-20, 20, -10, 10])
    plt.autoscale(False)
    cmap = mpl.cm.magma
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Probability')
    plt.xlabel('Z-Position (m)')
    plt.ylabel('X-Position (m)')
    plt.title('Probability of Collision for n=%s Action Sequences' % label)
    plt.scatter(GOAL[1], GOAL[0], s=20, c='red', marker="o")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()
    plt.pause(0.05)


def plot_action_cam_view(actions, frame, event_probs, steer_angle, current_speed):
    plt.cla()
    # plt.figure(figsize=(16, 8.9), dpi=80)
    # plt.rcParams["figure.figsize"] = (25, 25)
    pos = torch.zeros(actions.shape)
    dt = 0.1
    wb = 0.38
    omega = torch.tensor([-dt/2 * steer_angle * current_speed / wb])
    for i, val in enumerate(pos.transpose(1, 0)):
        if i+1 == pos.shape[1]:
            break
        pos[:, i+1, 0] = pos[:, i, 0] - dt * torch.sin(omega) * actions[:, i, 0]
        pos[:, i+1, 1] = pos[:, i, 1] + dt * torch.cos(omega) * actions[:, i, 0]
        omega = omega - dt * actions[:, i, 1] * actions[:, i, 0] / wb
    '''rescale x,y axes to look more 3D'''
    pos[:, :, 1] = pos[:, :, 1]/abs(pos[:, :, 1]).max()*1.2
    pos[:, :, 0] = pos[:, :, 0]/abs(pos[:, :, 0]).max()*0.8

    for p, e in zip(pos.numpy(), event_probs.numpy()):
        points = np.expand_dims(np.array([p[:, 0], p[:, 1]]).T, 1)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap('YlOrRd'), norm=plt.Normalize(0, 1))
        lc.set_array(e.T)
        lc.set_linewidth(1)
        plt.gca().add_collection(lc)
    plt.imshow(frame.int().numpy().transpose(1, 2, 0), extent=[-1.5, 1.5, 0, 2])
    plt.autoscale(False)
    plt.title('Probabilities of Unsafe Position')
    # cmap = mpl.cm.magma
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    # plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Probability')
    # plt.show()
    # plt.pause(0.05)


def count_data_ratio(df):
    print(len(df)*20/sum([torch.count_nonzero(row['Ground_Truth']) for index, row in df.iterrows()]))
    pass


if __name__ == "__main__":
    dir_name = Path(Path.cwd()).parent
    dir_name = str(dir_name) + '/Test/controllers/Hircus/'
    # df = pd.read_pickle(dir_name + "State_rewards.pkl")
    df = pd.read_pickle(dir_name + "Herdr_data_train.pkl")
    count_data_ratio(df)
    # for index, row in df.iterrows():
    #     plot_actions(row['State'], row['Event_Prob'], "50", row['Target_Pos'], dir_name)

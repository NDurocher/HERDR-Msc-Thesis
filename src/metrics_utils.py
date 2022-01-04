import matplotlib.pyplot
import torch
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from time import sleep


def plot_trajectory(robot_traj, line_values, ped_traj, goal, label, traj_length, collision=-1):
    # fig, ax = plt.subplots()
    points = np.array([robot_traj[:, 2], robot_traj[:, 0]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('magma'), norm=plt.Normalize(0, line_values.max()))
    lc.set_array(line_values)
    lc.set_linewidth(3)
    plt.figure(1)
    plt.gca().add_collection(lc)
    img = plt.imread("Topview.jpg")  #/Users/NathanDurocher/Documents/GitHub/HERDR/
    plt.imshow(img, extent=[-20, 20, -10, 10])
    plt.autoscale(False)
    plt.scatter(goal[0, 1], goal[0, 0], s=200, c='green', marker="o")
    for i, ped in enumerate(ped_traj):
        if i == 0:
            plt.scatter(ped[:, 2], ped[:, 0], s=0.8, c='orange', marker="o")
        else:
            plt.scatter(ped[:, 2], ped[:, 0], s=0.2, c='red', marker="o")
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
    plt.show()

def plot_actions(position, line_values, label, GOAL):
    plt.clf()
    for pos, lv in zip(position.transpose(1, 0, 2), line_values):
        points = np.expand_dims(np.array([pos[:, 2], pos[:, 0]]).T, 1)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap('magma'), norm=plt.Normalize(0, 1))
        lc.set_array(lv.T)
        lc.set_linewidth(3)
        plt.gca().add_collection(lc)
    plt.axis('equal')
    cmap = mpl.cm.magma
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Probability')
    plt.xlabel('Z-Position (m)')
    plt.ylabel('X-Position (m)')
    plt.title('Probability of Collision for n=%s Action Sequences' % label)
    # plt.scatter(GOAL[0, 1], GOAL[0, 0], s=50, c='red', marker="o")
    plt.show()
    # sleep(0.1)
    plt.pause(0.01)




import torch
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from time import sleep


def plot_trajectory(traj, line_values, goal, label, traj_length, collision=-1):
    points = np.array([traj[:, 2], traj[:, 0]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('magma'), norm=plt.Normalize(0, line_values.max()))
    lc.set_array(line_values)
    lc.set_linewidth(3)
    plt.gca().add_collection(lc)
    plt.scatter(goal[0, 1], goal[0, 0], s=600, c='red', marker="o")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axis('equal')
    plt.colorbar(mappable=lc, label='%s from Peds (m)' % label)
    plt.xlabel('Z-Position (m)')
    plt.ylabel('X-Position (m)')
    plt.title('%s from Pedestrians along a Trajectory' % label)
    if collision != -1:
        plt.figtext(0.15, 0.17, "# of Collisions %d" % collision)
    plt.figtext(0.15, 0.13, "Distance Travelled: %2.2f" % traj_length)
    plt.show()


def plot_actions(position, line_values, label):
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
    plt.show()
    # sleep(0.1)
    plt.pause(0.01)




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
import h5py
import shutil

def plot_trajectory(robot_traj, line_values, goal, traj_length=-1, collision=-1):
    plt.clf()
    plt.figure(figsize=(16, 16), dpi=80)
    points = np.array([robot_traj[:, 1], robot_traj[:, 0]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('cool'), norm=plt.Normalize(0, line_values.max()))
    lc.set_array(line_values)
    lc.set_linewidth(3)
    plt.gca().add_collection(lc)
    img = plt.imread("/home/nathan/HERDR/tsne/City_top_down.png")  #/Users/NathanDurocher/Documents/GitHub/HERDR/
    plt.imshow(img, extent=[-150, 150, -150, 150])
    plt.autoscale(False)
    plt.scatter(goal[1], goal[0], s=200, c='red', marker="o")
    start = robot_traj[0]
    plt.scatter(start[1], start[0], s=100, c='green', marker="o")
    # plt.axis('equal')
    robot_traj_array = np.asarray(robot_traj)
    xmin, xmax = max([min([robot_traj_array[:,1].min() -50, goal[1]-50]),-200]), min([max([robot_traj_array[:,1].max()+50,goal[1]+50]),200])
    ymin, ymax = max([min([robot_traj_array[:,0].min() -50, goal[0]-50]),-200]), min([max([robot_traj_array[:,0].max()+50,goal[0]+50]),200])
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
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
    plt.scatter(GOAL[1], GOAL[0], s=30, c='cyan', marker="o")


def plot_action_cam_view(position, frame, event_probs, state, planner_mean=None):
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
    # if type(planner_mean) == type(None):
    #     return
    # planner_mean = planner_mean.numpy().T
    # ''' Display optimal path from planner mean '''
    # opt_state = np.zeros((planner_mean.shape[0],3))
    # dt = 1/5
    # wb = 0.7
    # ''' opt_state := [X, Y, phi] '''
    # for i in range(0,len(planner_mean) - 1):
    #     opt_state[i + 1, 0] = opt_state[i, 0] + dt * np.cos(
    #         opt_state[i, 2]) * planner_mean[i, 0]
    #     opt_state[i + 1, 1] = opt_state[i, 1] + dt * np.sin(
    #         opt_state[i, 2]) * planner_mean[i, 0]
    #     opt_state[i + 1, 2] = opt_state[i, 2] - dt * planner_mean[i, 1] * planner_mean[i, 0] / wb
    # opt_state[:, 1] = opt_state[:, 1]/abs(opt_state[:, 1]).max()*1.1
    # opt_state[:, 0] = opt_state[:, 0]/abs(opt_state[:, 0]).max()*0.7 
    # points = np.expand_dims(np.array([opt_state[:, 1], opt_state[:, 0]]).T, 1)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # lc = LineCollection(segments, cmap=plt.get_cmap('YlGn'), norm=plt.Normalize(0, 1))
    # lc.set_array(np.ones((10)))
    # lc.set_linewidth(3)
    # plt.gca().add_collection(lc)



def count_data_ratio(loader):
    total = len(loader)*10
    positive = sum([torch.count_nonzero(gnd) for im, act, gnd in loader])
    ratio = total/positive
    print(f'Total Samples: {total}, # Positive: {positive} Ratio of total:positive {ratio:.4f}')
    return ratio

    
def moveimages(h5file_path, recursive=False):
    load_all_files = True
    if load_all_files:
            ''' Search for all h5 files else just load one'''
            p = Path(h5file_path)
            assert (p.is_dir())
            if recursive:
                files = sorted(p.glob('**/*.h5'))
            else:
                files = sorted(p.glob('*.h5'))
            if len(files) < 1:
                raise RuntimeError('No hdf5 datasets found')    
            data = np.concatenate([loadfromfile(str(h5dataset_fp.resolve())) for h5dataset_fp in files])
    else:
        data = loadfromfile(h5file_path)
 
        # '''find all indices where not done'''
        # valid_start_indices = np.where(data[:,-1] == 'False')[0]

    src = "/home/nathan/HERDR/carla_images/"
    dst = "/home/nathan/HERDR/carla_images_used/"

    for f in data[:,3]:
        shutil.move(os.path.join(src, f+".jpg"), dst)

def loadfromfile(file_path):
    with h5py.File(file_path) as h5_file:
        file_arr = np.ndarray(shape=(1,5))
        for gpname, gp in h5_file.items():
            # num_samples = len(gp['gnd'])
            group_arr = np.concatenate([gp[name][...].astype(str) if 'actions' in name else gp[name][...][:,None].astype(str) for name in gp.keys()], axis=1)
            done_arr = np.array([False if i < len(group_arr)-1 else True for i in range(len(group_arr))])
            group_arr = np.concatenate((group_arr, done_arr[:,None]), axis=1)
            file_arr = np.concatenate((file_arr, group_arr), axis=0)

    ''' Delete empty row at start of arr'''
    file_arr = file_arr[1:]
    
    ''' file array shape: [velocity, steer angle, gnd, image name, done]'''
    return file_arr

if __name__ == "__main__":
    dir_name = Path(Path.cwd())
    dir_name = str(dir_name) + '/all_carla_hdf5s/'
    dataset = carla_hdf5dataclass(dir_name, 10, '/home/nathan/HERDR/carla_images', counting=True, load_all_files=True, recursive=True)
    print(len(dataset))
    test_sampler = SubsetRandomSampler(dataset.valid_start_indices)
    testloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, batch_size=1)
    count_data_ratio(testloader)
    dataset.counting=False
    # testloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, batch_size=1)
    # print(next(iter(testloader)))
    # moveimages(dir_name,recursive=True)
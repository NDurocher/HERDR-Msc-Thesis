import numpy as np
from scipy import rand
from sklearn.manifold import TSNE
import torch
import glob
from Badgrnet import HERDR
from torchvision.io import read_image
from torchvision.transforms.functional import resize
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import random
import h5py
import cv2
from pathlib import Path
from Carla_Trainer import carla_hdf5dataclass
from torch.utils.data.sampler import SubsetRandomSampler


def impre(im, loader):
    im = loader(im).float()
    im = im.unsqueeze(0)
    return im

def calculate_position_angle(actions):
        state = torch.zeros((3))
        control_freq = 1/5
        wheelbase = 0.7
        ''' [X Y Phi] '''
        for i in range(0, 9):
            state[0] = state[0] + (1/control_freq) * torch.cos(state[2]) * actions[0, i, 0]
            state[1] = state[1] + (1/control_freq) * torch.sin(state[2]) * actions[0, i, 0]
            state[2] = state[2] - (1/control_freq) * actions[0, i, 1] * actions[0, i, 0] / wheelbase
        ''' Output shape: [1, 3] '''
        angle = torch.atan2(state[1],state[0])
        return angle

def plot(data, tsne_results, model_name, ncolours):
    data['tsne-2d-one'] = tsne_results[:, 0]
    data['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 9), dpi=90)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="key",
        palette=sns.color_palette("bright", ncolours),
        data=data,
        alpha=0.3
    )
    plt.title(f'TSNE of Embedded Vectors After Layers: {model_name}')


def keyMaker(df_emb, df_key, value_name):
    if 'steer' in value_name:
        bins = np.arange(-0.95, 0.95, 0.20).tolist()
        labels = np.linspace(-0.95, 0.95, 9).tolist()
        df_emb['key'] = pd.cut(df_key[0], bins=bins, labels=labels)
        ncolors = len(labels)
        return df_emb, ncolors
    if 'angles' in value_name:
        bins = np.arange(-3.14, 3.14, 0.63).tolist()
        labels = np.linspace(-3.14, 3.14, 9).tolist()
        df_emb['key'] = pd.cut(df_key[0], bins=bins, labels=labels)
        ncolors = len(labels)
        return df_emb, ncolors
    if 'gnd' in value_name:
        df_emb['key'] = df_key.loc[:,0]
        ncolors = len(np.unique(df_emb['key']))
        return df_emb, ncolors
        

def act_checker(model, model_name, recursive=False):
    model = model.action_pre
    var = np.concatenate([np.expand_dims(np.linspace(x, 0, 10),1) for x in np.linspace(-0.95, 0.95, 200)], axis=1)
    speed = np.ones((10,200,1))
    actions = np.concatenate((speed,var[:,:, None]), axis=2)
    actions_emb = [model(torch.tensor(act).to(device).float()).detach().squeeze(0).cpu().numpy() for act in actions.transpose(1,0,2)]
    angles = np.asarray([calculate_position_angle(act[None,:,:]) for act in actions.transpose(1,0,2)])

    ''' Key for steering angles is var.T, for angles it's '''
    for a_idx in range(np.asarray(actions_emb).shape[1]):
        df_key = pd.DataFrame(var.T)
        # df_key = pd.DataFrame(angles)
        df_emb = pd.DataFrame(np.asarray(actions_emb)[:,a_idx,:])
        df_emb, ncolors = keyMaker(df_emb, df_key, 'steer')
        Tnse_results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df_emb.loc[:, 0:15])
        plot(df_emb, Tnse_results, model_name, ncolors)
        plt.savefig(f'/home/nathan/HERDR/tsne/plots/tsne-actions{a_idx}.png')


def cnn_check(model, model_name):
    model = model.obs_pre
    w_ped = [resize(read_image(file).unsqueeze(0).float(), [480, 640]) for file in glob.glob("/home/nathan/HERDR/tsne/with_ped/*.png")]
    wo_ped = [resize(read_image(file).unsqueeze(0).float(), [480, 640]) for file in glob.glob("/home/nathan/HERDR/tsne/without_ped/*.png")]
    w_ped_emb = [model(im.to(device)).detach().squeeze(0).numpy() for im in w_ped]
    wo_ped_emb = [model(im.to(device)).detach().squeeze(0).numpy() for im in wo_ped]
    key_w = ['with Pedestrians'] * len(w_ped)  # or 0
    key_wo = ['Without Pedestrians'] * len(wo_ped)  # or 1
    df_w = pd.DataFrame(w_ped_emb)
    df_w['key'] = key_w
    df_wo = pd.DataFrame(wo_ped_emb)
    df_wo['key'] = key_wo

    df_tot = pd.DataFrame()
    df_tot = df_tot.append(df_w)
    df_tot = df_tot.append(df_wo)
    df_tot['key'] = df_tot['key'].astype("category")
    Tnse_results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df_tot.loc[:, 0:127])
    plot(df_tot, Tnse_results, model_name, 2)
    plt.show()


def check_lstm(model, model_name, recursive=False):
    model = lstm(model)
    
    # dataset = carla_hdf5dataclass('/home/nathan/HERDR/tsne/h5', 10, imagefile_path='/home/nathan/HERDR/tsne/images', load_all_files=True)
    dataset = carla_hdf5dataclass('/home/nathan/HERDR/old_carla_hdf5s/20-04-2022--19-38', 10, imagefile_path='/home/nathan/HERDR/old_carla_images/20-04-2022--19-38/', load_all_files=True)
    test_sampler = SubsetRandomSampler(dataset.valid_start_indices)
    testloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, batch_size=1)
    vec_emb = []
    all_actions = []
    all_gnd = []
    all_angles = []
    img, _, _ = next(iter(testloader))
    cv2.imwrite(f'/home/nathan/HERDR/tsne/plots/lstm_steer_sweep.jpg', img.squeeze(0).permute(1, 2, 0).numpy())
    for _, act, gnd in testloader:
        vec_emb.append(model(img.to(device),act.to(device)).detach().cpu().squeeze(0).numpy())
        all_actions.append(act.numpy())
        all_gnd.append(gnd.numpy())
        all_angles.append(calculate_position_angle(act))
    for a_idx in range(np.asarray(vec_emb).shape[1]):
        ''' 2ndth index is which time step will be used as label '''
        # df_key = pd.DataFrame(np.asarray(all_actions)[:,0,a_idx,1])
        # df_key = pd.DataFrame(np.asarray(all_gnd)[:,0,a_idx,0])
        df_key = pd.DataFrame(np.asarray(all_angles))
        df_emb = pd.DataFrame(np.asarray(vec_emb)[:,a_idx, 0, :].tolist())
        df_emb, ncolors = keyMaker(df_emb, df_key, 'angles')
        Tnse_results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df_emb.loc[:, 0:63])
        plot(df_emb, Tnse_results, model_name, ncolors)
        plt.savefig(f'/home/nathan/HERDR/tsne/plots/tsne-lstm{a_idx}.png')

def check_lstm_one_img_many_st_angles(model,model_name):
    model = lstm(model)
    dataset = carla_hdf5dataclass('/home/nathan/HERDR/tsne/h5', 10, imagefile_path='/home/nathan/HERDR/tsne/images', load_all_files=True)
    test_sampler = SubsetRandomSampler(dataset.valid_start_indices)
    testloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, batch_size=1)
    # img = read_image(f"/home/nathan/HERDR/tsne/images/2022-03-31 20:46:36.848480.jpg").float().unsqueeze(0)
    img, _, _ = next(iter(testloader))
    var = np.concatenate([np.expand_dims(np.linspace(x, 0, 10),1) for x in np.linspace(-0.95, 0.95, 200)], axis=1)
    speed = np.ones((10,1))
    actions_emb = [model(img.to(device),torch.tensor(np.concatenate((speed,np.expand_dims(action,1)),axis=1)).to(device).float().unsqueeze(0)).detach().squeeze(0).cpu().numpy() for action in var.T]
    
    for a_idx in range(np.asarray(actions_emb).shape[1]):
        # df_key = pd.DataFrame(np.asarray(all_angles))
        df_key = pd.DataFrame(var.T)
        df_emb = pd.DataFrame(np.asarray(actions_emb)[:, a_idx, 0, :].tolist())
        df_emb, ncolors = keyMaker(df_emb, df_key, 'steer')
        Tnse_results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df_emb.loc[:, 0:63])
        plot(df_emb, Tnse_results, model_name, ncolors)
        plt.savefig(f'/home/nathan/HERDR/tsne/plots/tsne-lstm{a_idx}.png')
    cv2.imwrite(f'/home/nathan/HERDR/tsne/plots/lstm_steer_sweep.jpg', img.squeeze(0).permute(1, 2, 0).numpy())

class lstm(HERDR):
    def __init__(self, model):
        super().__init__()
        self.horizon = model.horizon
        self.rnndim = model.rnndim
        self.lstm = model.lstm
        self.obs_pre = model.obs_pre
        self.init_hidden = model.init_hidden
        self.action_pre = model.action_pre

    def forward(self, img, action):
        obs = self.obs_pre(self.normalize(img))
        # Change obs to 2*rnndim encoding, this is then split into Hx and Cx
        obs = self.init_hidden(obs)
        Hx, Cx = torch.chunk(obs, 2, dim=1)
        Hx = Hx.repeat(1, 1, 1)
        Cx = Cx.repeat(1, 1, 1)
        action = self.action_pre(action)
        # put "time" first
        action = action.transpose(1, 0)
        out, (_, _) = self.lstm(action, (Hx, Cx))
        
        ''' Below outputs the cell state at each timestep'''
        # for i, act in enumerate(action):
        #     out, (Hx, Cx) = self.lstm(act.unsqueeze(1), (Hx, Cx))
        #     if i == 0:
        #         c_out = Cx.squeeze(0)
        #     else:
        #         c_out = torch.cat((c_out,Cx.squeeze(0)), dim=0)
        # return c_out
        return out


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Use GPU")
    else:
        device = torch.device('cpu')
        print("Use CPU")
    # torch.manual_seed(12)
    modl_name = "carla22-04-2022--09:48.pth"
    modl = torch.load(f"/home/nathan/HERDR/models/{modl_name}")
    # cnn_check(modl, modl_name)
    act_checker(modl, modl_name)
    # check_lstm(modl, modl_name)
    # check_lstm_one_img_many_st_angles(modl, modl_name)


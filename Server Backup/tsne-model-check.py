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


def plot(data, tsne_results, model_name, ncolours):
    data['tsne-2d-one'] = tsne_results[:, 0]
    data['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 9), dpi=90)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="key",
        palette=sns.color_palette("deep", ncolours),
        data=data,
        alpha=0.3
    )
    plt.title(f'TSNE of Embedded Vectors After Layers: {model_name}')


def act_checker(model, model_name, recursive=False):
    model = model.action_pre

    # p = Path("/home/nathan/HERDR/tsne/h5")
    # assert (p.is_dir())
    # if recursive:
    #     files = sorted(p.glob('**/*.h5'))
    # else:
    #     files = sorted(p.glob('*.h5'))
    # if len(files) < 1:
    #     raise RuntimeError('No hdf5 datasets found')

    # all_actions = []
    # for file_path in files:
    #     with h5py.File(file_path) as h5_file:
    #         for num_gp, (gname, group) in enumerate(h5_file.items()):
    #             if num_gp > 4:
    #                 break
    #             for dname, ds in group.items():
    #                 if dname == "actions":
    #                     for batch in ds[...]:
    #                         all_actions.append(batch)
    # all_actions = np.asarray(all_actions)
    var = np.concatenate([np.expand_dims(np.linspace(x, 0, 10),1) for x in np.linspace(-0.95, 0.95, 200)], axis=1)
    speed = np.ones((10,1))
    actions_emb = [model(torch.tensor(np.concatenate((speed,np.expand_dims(act,1)),axis=1)).to(device).float()).detach().squeeze(0).cpu().numpy() for act in var.T]

    for a_idx in range(np.asarray(actions_emb).shape[1]):
        #  Take all time=a_idx actions
    # indices = random.choices(np.arange(len(all_actions)), k=200)
        df_steer = pd.DataFrame(var.T)
        bins = np.arange(-0.95, 0.95, 0.20).tolist()
        labels = np.linspace(-0.95, 0.95, 9).tolist()
        df_emb_steer = pd.DataFrame(np.asarray(actions_emb)[:,a_idx,:])
        df_emb_steer['key'] = pd.cut(df_steer[0], bins=bins, labels=labels)
        Tnse_results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df_emb_steer.loc[:, 0:15])
        plot(df_emb_steer, Tnse_results, model_name, 9)
        plt.savefig(f'/home/nathan/HERDR/tsne/plots/tsne-actions{a_idx}.png')


def main(model, model_name):
    model = model.obs_pre
    w_ped = [resize(read_image(file).unsqueeze(0).float(), [720, 1280]) for file in glob.glob("../Test/controllers/Hircus/tnse_ped_other/*.png")]
    wo_ped = [resize(read_image(file).unsqueeze(0).float(), [720, 1280]) for file in glob.glob("../Test/controllers/Hircus/tnse_no_ped/*.png")]
    w_ped_emb = [model(im.to(device)).detach().squeeze(0).numpy() for im in w_ped]
    wo_ped_emb = [model(im.to(device)).detach().squeeze(0).numpy() for im in wo_ped]
    key_w = ['with Pedestrians'] * len(w_ped)  # or close to
    key_wo = ['Without Pedestrians'] * len(wo_ped)  # or far away
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
    
    dataset = carla_hdf5dataclass('/home/nathan/HERDR/tsne/h5', 10, imagefile_path='/home/nathan/HERDR/tsne/images', load_all_files=True)
    test_sampler = SubsetRandomSampler(dataset.valid_start_indices)
    testloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, batch_size=1)
    vec_emb = []
    all_actions = []
    for img, act, gnd in testloader:
        vec_emb.append(model(img.to(device),act.to(device)).detach().cpu().squeeze(0).numpy())
        all_actions.append(act.numpy())
    for a_idx in range(np.asarray(vec_emb).shape[1]):
        #  Take all time=a_idx actions
        # indices = random.choices(np.arange(len(all_actions)), k=200)
        df_steer = pd.DataFrame(np.asarray(all_actions)[:,0,a_idx,1])
        # df_gnd = pd.DataFrame(all_gnd[:, a_idx])
        bins = np.arange(-0.95, 0.95, 0.20).tolist()
        labels = np.linspace(-0.95, 0.95, 9).tolist()
        df_emb_steer = pd.DataFrame(np.asarray(vec_emb)[:,a_idx, 0, :].tolist())
        df_emb_steer['key'] = pd.cut(df_steer[0], bins=bins, labels=labels)
        # df_emb_steer['key'] = df_gnd.loc[:,0]
        Tnse_results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df_emb_steer.loc[:, 0:63])
        # ncolors = np.unique(all_gnd[:, a_idx]).shape[0]
        ncolors = len(labels)
        plot(df_emb_steer, Tnse_results, model_name, ncolors)
        plt.savefig(f'/home/nathan/HERDR/tsne/plots/tsne-lstm{a_idx}.png')

def check_lstm_one_img_many_st_angles(model,model_name):
    model = lstm(model)
    dataset = carla_hdf5dataclass('/home/nathan/HERDR/tsne/h5', 10, imagefile_path='/home/nathan/HERDR/tsne/images', load_all_files=True)
    test_sampler = SubsetRandomSampler(dataset.valid_start_indices)
    testloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, batch_size=1)
    img = read_image(f"/home/nathan/HERDR/tsne/images/2022-03-31 20:46:36.848480.jpg").float().unsqueeze(0)
    var = np.concatenate([np.expand_dims(np.linspace(x, 0, 10),1) for x in np.linspace(-0.95, 0.95, 200)], axis=1)
    speed = np.ones((10,1))
    # img, act, gnd = next(iter(testloader))
    actions_emb = [model(img.to(device),torch.tensor(np.concatenate((speed,np.expand_dims(action,1)),axis=1)).to(device).float().unsqueeze(0)).detach().squeeze(0).cpu().numpy() for action in var.T]
    
    for a_idx in range(np.asarray(actions_emb).shape[1]):
        #  Take all time=a_idx actions
        # indices = random.choices(np.arange(len(all_actions)), k=200)
        df_steer = pd.DataFrame(var.T)
        # df_gnd = pd.DataFrame(all_gnd[:, a_idx])
        bins = np.arange(-0.95, 0.95, 0.20).tolist()
        labels = np.linspace(-0.95, 0.95, 9).tolist()
        df_emb_steer = pd.DataFrame(np.asarray(actions_emb)[:, a_idx, 0, :].tolist())
        df_emb_steer['key'] = pd.cut(df_steer[0], bins=bins, labels=labels)
        # df_emb_steer['key'] = df_gnd.loc[:,0]
        Tnse_results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df_emb_steer.loc[:, 0:63])
        # ncolors = np.unique(all_gnd[:, a_idx]).shape[0]
        ncolors = len(labels)
        plot(df_emb_steer, Tnse_results, model_name, ncolors)
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
        return out


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Use GPU")
    else:
        device = torch.device('cpu')
        print("Use CPU")
    torch.manual_seed(12)
    modl_name = "carla04-04-2022--14:03.pth"
    modl = torch.load(f"/home/nathan/HERDR/models/{modl_name}")
    # main(modl, modl_name)
    # act_checker(modl, modl_name)
    # check_lstm(modl, modl_name)
    check_lstm_one_img_many_st_angles(modl, modl_name)


import numpy as np
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
import h5py
from pathlib import Path


def impre(im, loader):
    im = loader(im).float()
    im = im.unsqueeze(0)
    return im


def plot(data, tsne_results, model_name, ncolours):
    data['tsne-2d-one'] = tsne_results[:, 0]
    data['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(8, 5))
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

    p = Path("/Users/NathanDurocher/Documents/GitHub/HERDR/src/h5")
    assert (p.is_dir())
    if recursive:
        files = sorted(p.glob('**/*.h5'))
    else:
        files = sorted(p.glob('*.h5'))
    if len(files) < 1:
        raise RuntimeError('No hdf5 datasets found')

    all_actions = []
    for file_path in files:
        with h5py.File(file_path) as h5_file:
            for num_gp, (gname, group) in enumerate(h5_file.items()):
                if num_gp > 4:
                    break
                for dname, ds in group.items():
                    if dname == "actions":
                        for batch in ds[...]:
                            all_actions.append(batch)
    all_actions = np.asarray(all_actions)
    actions_emb = [model(torch.tensor(act)).detach().squeeze(0).numpy() for act in all_actions]

    for a_idx in range(np.asarray(actions_emb).shape[1]):
        #  Take all time=a_idx actions
        df_steer = pd.DataFrame(all_actions[0:200, a_idx, 1])
        bins = np.arange(-0.95, 0.95, 0.20).tolist()
        labels = np.arange(9).tolist()
        df_emb_steer = pd.DataFrame(np.asarray(actions_emb)[0:200, a_idx, :].tolist())
        df_emb_steer['key'] = pd.cut(df_steer[0], bins=bins, labels=labels)
        Tnse_results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df_emb_steer.loc[:, 0:15])
        plot(df_emb_steer, Tnse_results, model_name, 9)
        plt.show()


def main(model, model_name):
    model = model.obs_pre
    w_ped = [resize(read_image(file).unsqueeze(0).float(), [720, 1280]) for file in glob.glob("../Test/controllers/Hircus/tnse_ped_other/*.png")]
    wo_ped = [resize(read_image(file).unsqueeze(0).float(), [720, 1280]) for file in glob.glob("../Test/controllers/Hircus/tnse_no_ped/*.png")]
    w_ped_emb = [model(im).detach().squeeze(0).numpy() for im in w_ped]
    wo_ped_emb = [model(im).detach().squeeze(0).numpy() for im in wo_ped]
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

    p = Path("/Users/NathanDurocher/Documents/GitHub/HERDR/src/h5")
    assert (p.is_dir())
    if recursive:
        files = sorted(p.glob('**/*.h5'))
    else:
        files = sorted(p.glob('*.h5'))
    if len(files) < 1:
        raise RuntimeError('No hdf5 datasets found')

    all_actions = []
    all_img = []
    all_gnd = []
    for file_path in files:
        with h5py.File(file_path) as h5_file:
            for num_gp, (gname, group) in enumerate(h5_file.items()):
                if num_gp > 4:
                    break
                for dname, ds in group.items():
                    for batch in ds[...]:
                        if dname == "actions":
                            all_actions.append(batch)
                        elif dname == "img":
                            all_img.append(batch)
                        elif dname == "gnd":
                            all_gnd.append(batch)
    all_actions = np.asarray(all_actions)
    all_gnd = np.asarray(all_gnd)
    vec_emb = [model(read_image(f'images/{img.decode("UTF-8")}.jpg').unsqueeze(0).float(),
                     torch.tensor(act).unsqueeze(0)).detach().squeeze(0).numpy() for img, act in zip(all_img, all_actions)]
    for a_idx in range(np.asarray(vec_emb).shape[1]):
        #  Take all time=a_idx actions
        # df_steer = pd.DataFrame(all_actions[:, a_idx, 1])
        df_gnd = pd.DataFrame(all_gnd[:, a_idx])
        # bins = np.arange(-0.95, 0.95, 0.20).tolist()
        # labels = np.arange(9).tolist()
        df_emb_steer = pd.DataFrame(np.asarray(vec_emb)[:, a_idx, 0, :].tolist())
        # df_emb_steer['key'] = pd.cut(df_steer[0], bins=bins, labels=labels)
        df_emb_steer['key'] = df_gnd.loc[:,0]
        Tnse_results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df_emb_steer.loc[:, 0:63])
        ncolors = np.unique(all_gnd[:, a_idx]).shape[0]
        plot(df_emb_steer, Tnse_results, model_name, ncolors)
        plt.show()


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
    modl_name = "Herdr_Best_Feb22.pth"
    modl = torch.load(f"../Test/controllers/Hircus/{modl_name}",
                       map_location=torch.device('cpu'))
    # main(modl, modl_name)
    # act_checker(modl, modl_name)
    check_lstm(modl, modl_name)
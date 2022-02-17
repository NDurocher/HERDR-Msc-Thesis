import numpy as np
from sklearn.manifold import TSNE
from torch import nn
import torch
import cv2
import glob
from Badgrnet import HERDR
from torchvision import transforms
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torchvision


def impre(im, loader):
    im = loader(im).float()
    im = im.unsqueeze(0)
    return im

def plot(data, tsne_results, model_name):
    data['tsne-2d-one'] = tsne_results[:, 0]
    data['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="key",
        palette=sns.color_palette("deep", 2),
        data=data,
        alpha=0.3
    )
    plt.title(f'TSNE of Embedded Vectors after HERDR Convolutional Layers: {model_name}')


def main():
    model_name = "Herdr17-02-2022--10 54 25.pth"
    model = torch.load(f"../Test/controllers/Hircus/{model_name}",
                       map_location=torch.device('cpu'))
    model = model.obs_pre
    w_ped = [torchvision.io.read_image(file).unsqueeze(0).float() for file in glob.glob("../Test/controllers/Hircus/tnse_ped/*.png")]
    wo_ped = [torchvision.io.read_image(file).unsqueeze(0).float() for file in glob.glob("../Test/controllers/Hircus/tnse_no_ped/*.png")]
    w_ped_emb = [model(im).detach().squeeze(0).numpy() for im in w_ped]
    wo_ped_emb = [model(im).detach().squeeze(0).numpy() for im in wo_ped]
    key_w = ['With Pedestrians'] * len(w_ped)
    key_wo = ['Without Pedestrians'] * len(wo_ped)
    df_w = pd.DataFrame(w_ped_emb)
    df_w['key'] = key_w
    df_wo = pd.DataFrame(wo_ped_emb)
    df_wo['key'] = key_wo

    df_tot = pd.DataFrame()
    df_tot = df_tot.append(df_w)
    df_tot = df_tot.append(df_wo)
    df_tot['key'] = df_tot['key'].astype("category")
    Tnse_results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df_tot.loc[:, 0:127])
    plot(df_tot, Tnse_results, model_name)
    plt.show()


if __name__ == "__main__":
    main()
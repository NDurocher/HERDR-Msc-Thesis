"""Logger controller."""

from controller import Supervisor
import numpy as np
import pickle
import torch
# from torch import nn
from torchvision.utils import save_image
from pathlib import Path
import sys
from matplotlib import pyplot as plt
import matplotlib.animation as animation

dir_name = Path(Path.cwd()).parent.parent.parent
sys.path.insert(1, str(dir_name)+'/src')

from metrics_utils import plot_trajectory


class Logger (Supervisor):
    def __init__(self):
        """Constructor: initialize constants."""
        Supervisor.__init__(self)
        self.cc = self.getSelf()
        self.customdata = self.cc.getField('customData')
        self.timestep = int(self.getBasicTimeStep())
        self.hircus = self.getFromDef("Hircus")
        self.camera = self.getDevice('camera')
        self.camera.enable(self.timestep)
        self.height = self.camera.getHeight()
        self.width = self.camera.getWidth()
        self.peds = []
        self.frame = None
        i = 0
        while not self.getFromDef("Ped%d" % i) is None:
            self.peds.append(self.getFromDef("Ped%d" % i))
            i += 1

    @staticmethod
    def log(hircuspos, pedlist):
        dist_list = []
        collision = 0
        for person in pedlist:
            pos = person.getPosition()
            dist = np.sqrt((hircuspos[0]-pos[0])**2 + (hircuspos[2]-pos[2])**2)
            dist_list.append(dist)
            if dist < 0.5:
                collision = 1
        try:
            min_dist = np.asarray(dist_list).min()
        except:
            min_dist = 0.
        return np.asarray([min_dist, collision])

    def grab_frame(self):
        frame = np.asarray(np.frombuffer(self.camera.getImage(), dtype=np.uint8))
        frame = np.reshape(np.ravel(frame), (self.height, self.width, 4), order='C')
        frame = torch.tensor(frame[:, :, 0:3]).float()
        frame = frame.permute(2, 0, 1)
        frame = torch.index_select(frame, 0, torch.tensor([2, 1, 0]))
        save_image(frame / 255, 'Topview.jpg')


def pathlength(x, y):
    n = len(x)
    lv = [np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) for i in range(1, n)]
    L = sum(lv)
    return L


def is_float(value):
    try:
        float(value)
        return float(value)
    except:
        return 0


Hircus_traj = []
ped_trajs = []
min_dist = []
in_collision = []
writer_dist = animation.FFMpegWriter(fps=7)
writer_score = animation.FFMpegWriter(fps=7)
fig = plt.figure(figsize=(16, 8.9), dpi=80)
writer_dist.setup(fig, '/home/nathan/HERDR/VideosOut/clearance.mp4')
writer_score.setup(fig, '/home/nathan/HERDR/VideosOut/score.mp4')
controller = Logger()
GOAL = controller.cc.getPosition()
unsafe_score = []
while not controller.step(controller.timestep) == -1:
    GOAL = controller.cc.getPosition()
    Hircuspos = controller.hircus.getPosition()
    Hircus_traj.append(np.asarray(Hircuspos-np.asarray(GOAL)))
    if controller.getTime() > 0.1:
        ped_trajs.append([p.getPosition() - np.asarray(GOAL) for p in controller.peds])
    out_log = controller.log(Hircuspos, controller.peds)
    min_dist.append(out_log[0])
    in_collision.append(out_log[1])
    # controller.exportImage(str(dir_name) + '/Test/controllers/Logger/Topview.jpg', 100)
    controller.grab_frame()
    traj_length = pathlength(np.asarray(Hircus_traj)[:, 0], np.asarray(Hircus_traj)[:, 2])
    unsafe_score.append(is_float(controller.customdata.getSFString()))
    plot_trajectory(np.asarray(Hircus_traj), np.asarray(min_dist), np.asarray(ped_trajs), np.array([0, 0, 0]),
                    "Clearance", traj_length, collision=in_collision.count(1))
    writer_dist.grab_frame()
    plot_trajectory(np.asarray(Hircus_traj), np.asarray(unsafe_score), np.asarray(ped_trajs), np.array([0, 0, 0]),
                    "Event Totals", traj_length, collision=in_collision.count(1))
    writer_score.grab_frame()
writer_dist.finish()
writer_score.finish()



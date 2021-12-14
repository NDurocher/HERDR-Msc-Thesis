#!*python*

# import numpy as np
from torch import nn
import torch
import cv2
from torchvision import transforms
from actionplanner import HERDRPlan


class HERDR(nn.Module):
    def __init__(self, Horizon=1, RnnDim=16):
        super().__init__()
        self.horizon = Horizon
        self.rnndim = RnnDim

        self.obs_pre = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(2, 2)),
            nn.MaxPool2d(4, stride=2),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.MaxPool2d(4, stride=2),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(256, 128),
            # nn.LazyBatchNorm1d()
        )
        self.action_pre = nn.Sequential(
            nn.Linear(2, 16),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.init_hidden = nn.Sequential(
            nn.Linear(128, 128),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(128, 2 * self.rnndim)
        )
        self.model_out = nn.Sequential(
            nn.Linear(self.rnndim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        self.lstm = nn.LSTM(input_size=16, hidden_size=self.rnndim, num_layers=1, batch_first=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, img, action):
        obs = self.obs_pre(img)
        # Change obs to 2*rnndim encoding, this is then split into Hx and Cx
        obs = self.init_hidden(obs)
        Hx, Cx = torch.chunk(obs, 2, dim=1)
        Hx = Hx.repeat(1, 1, 1)
        Cx = Cx.repeat(1, 1, 1)
        # put "time" first
        # action = action.transpose(2, 1)
        act = self.action_pre(action)
        out, (Hn, _) = self.lstm(act, (Hx, Cx))
         # put "time" first
        out = out.transpose(1, 0)
        out = self.model_out(out)
        # out = self.softmax(out)
        out = out.transpose(1, 0)
        # out = out.transpose(2, 1)
        # Output shape is (Batch, Horizon, 1))
        return out


if __name__ == "__main__":
    def impreprosses(im):
        im = cv2.resize(im, (320, 240))
        im = loader(im).float()
        im = im.unsqueeze(0)
        im = im.repeat(batches, 1, 1, 1)
        return im

    batches = 50
    hor = 20
    planner = HERDRPlan(Horizon=hor)
    model = HERDR(Horizon=hor, RnnDim=64)
    # model = torch.load("/Users/NathanDurocher/Documents/GitHub/HERDR/Test/controllers/Hircus/Herdr_5_LSTM5.pth",
    #                    map_location=torch.device('cpu'))
    video = cv2.VideoCapture(0)

    # Clear camera opening frame
    _, _ = video.read()

    # Take one image
    loader = transforms.Compose([transforms.ToTensor()])

    for i in range(0, 1):
    # while True:
        check, frame = video.read()
        frame = impreprosses(frame)
        actions = planner.sample_new(batches=batches)
        print(frame.shape, actions.shape)
        r = model(frame, actions)
        print(r.flatten(start_dim=0, end_dim=1).shape)

        # torch.onnx.export(model,(frame, actions),'Herdr.onnx')
        # print("Prediction: ", r.detach())
        # print(r.shape, actions.shape)
        # planner.update_new(r.detach(), actions)
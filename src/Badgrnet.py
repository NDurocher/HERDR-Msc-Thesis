#!*python*

# import numpy as np
from PIL import Image
from torch import nn
import torch
import cv2
from torchvision import transforms
from actionplanner import BadgrPlan


class BadgrNet(nn.Module):
    def __init__(self, Horizon=1, RnnDim=16, Batch=1):
        super().__init__()
        self.horizon = Horizon
        self.rnndim = RnnDim
        self.batch = Batch

        self.obs_pre = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.action_pre = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.init_hidden = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * self.rnndim)
        )
        self.model_out = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.lstm = nn.LSTM(input_size=16, hidden_size=self.rnndim, num_layers=1, batch_first=True)

    def forward(self, img, action):
        obs = self.obs_pre(img)
        # Change obs to 2*rnndim encoding, this is then split into Hx and Cx
        obs = self.init_hidden(obs)
        Hx, Cx = torch.chunk(obs, 2, dim=1)
        # Hx = Hx.repeat(self.horizon, 1, 1)
        # Cx = Cx.repeat(self.horizon, 1, 1)
        Hx = Hx.repeat(1, 1, 1)
        Cx = Cx.repeat(1, 1, 1)
        # put "time" first
        action = action.transpose(2, 1)
        act = self.action_pre(action)
        out, (_, _) = self.lstm(act, (Hx.detach(), Cx.detach()))
        # put "time" second
        out = self.model_out(out)
        print(out.shape)
        out = out.transpose(2, 1)
        # Output shape is (batch, 1, Horizon)
        return out


if __name__ == "__main__":  
    batches = 150
    hor = 5
    planner = BadgrPlan(Horizon=hor)
    model = BadgrNet(Horizon=hor, Batch=batches)
    # Herdr = torch.load("Herdr_1.pth", map_location=torch.device('cpu'))
    video = cv2.VideoCapture(0)
    check, frame = video.read()
    loader = transforms.Compose([transforms.ToTensor()])
    for i in range(0, 1):
    # while True:
        check, frame = video.read()
        frame = cv2.resize(frame, (320, 240))
        frame = loader(frame).float()
        frame = frame.unsqueeze(0)
        actions = planner.sample_new()
        # actions = actions.unsqueeze(0)
        prediction = model(frame, actions)
        torch.onnx.export(model,(frame, actions),'Herdr.onnx')
        print("Prediction: ", prediction.detach().numpy().ravel())
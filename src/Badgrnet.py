#!*python*

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
            # nn.LayerNorm([3, 240, 320]),
            nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(2, 2)),
            nn.MaxPool2d(4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.MaxPool2d(4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.action_pre = nn.Sequential(
            # nn.LayerNorm([20, 2]),
            nn.Linear(2, 16),
            # nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.init_hidden = nn.Sequential(
            nn.Linear(128, 128),
            # nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(128, 2 * self.rnndim)
        )
        self.model_out = nn.Sequential(
            nn.Linear(self.rnndim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Sigmoid()
        )
        self.lstm = nn.LSTM(input_size=16, hidden_size=self.rnndim, num_layers=1, batch_first=False)

    def normalize(self, arr):
        up, low = arr.max(), arr.min()
        mu = arr.mean()
        std = 0.5 * (up - low)
        normed_arr = (arr - mu) / std
        # arr = arr/255 - 0.5
        return normed_arr

    def forward(self, img, action):
        obs = self.obs_pre(self.normalize(img))
        # Change obs to 2*rnndim encoding, this is then split into Hx and Cx
        obs = self.init_hidden(obs)
        Hx, Cx = torch.chunk(obs, 2, dim=1)
        # # Hx = Hx.repeat(1,1,1)
        # # Cx = Cx.repeat(1,1,1)
        # Hx = Hx[None,:,:]
        # Cx = Cx[None,:,:]
        Hx = Hx.repeat(1, action.shape[0], 1)
        Cx = Cx.repeat(1, action.shape[0], 1)
        action = self.action_pre(action)
        # put "time" first
        action = action.transpose(1, 0)
        out, (_, _) = self.lstm(action, (Hx, Cx))
        # put "batch" first
        out = out.transpose(0, 1)
        out = self.model_out(out)
        # Output shape is (Batch, Horizon, 1))
        return out


if __name__ == "__main__":
    def impreprosses(im):
        im = cv2.resize(im, (1280, 720))
        im = loader(im).float()
        im = im.unsqueeze(0)
        im = im.repeat(batches, 1, 1, 1)
        return im

    batches = 25
    hor = 20
    planner = HERDRPlan(Horizon=hor)
    # model = HERDR(Horizon=hor, RnnDim=64)
    model = torch.load("/Users/NathanDurocher/Documents/GitHub/HERDR/Test/controllers/Hircus/Herdr_Best_Feb22.pth",
                       map_location=torch.device('cpu'))
    model.model_out = nn.Sequential(
        model.model_out,
        nn.Sigmoid()
    )
    model.eval()
    video = cv2.VideoCapture(0)

    # Clear camera opening frame
    _, _ = video.read()

    # Take one image
    loader = transforms.Compose([transforms.ToTensor()])
    t1 = torch.ones(hor, batches)
    # for i in range(0, 1):
    while True:
        check, frame = video.read()
        frame = impreprosses(frame)
        actions = planner.sample_new(batches=batches)
        # print(frame.shape, actions.shape)
        r = model(frame, actions)
        # print(r.flatten(start_dim=0, end_dim=1).shape)
        # tout = torch.count_nonzero(abs(r[:, :, 0] - t1) < 0.11)
        # print(tout)
        # torch.onnx.export(model,(frame, actions),'Herdr.onnx')
        print("Prediction: ", r.detach().flatten())
        break
        # print(r.shape, actions.shape)
        # planner.update_new(r.detach(), actions)
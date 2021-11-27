

import numpy as np
# from PIL import Image
# from torch import nn
import torch
# import cv2


class HERDRPlan:
    def __init__(self, Horizon=10, vel_init=1.5, steer_init=0, gamma=5):
        # Set default starting value for actions
        # [Velocity (m/s), Steering angle (rad)]
        self.horizon = Horizon
        vel_mean = vel_init*np.ones(self.horizon)
        steer_mean = steer_init*np.zeros(self.horizon)
        # self.mean = np.concatenate((vel_mean, steer_mean))
        self.mean = torch.tensor(np.vstack((vel_mean, steer_mean)))
        # set guess for variance
        vel_cov = 0.1*torch.ones(self.horizon, 1)
        steer_cov = 0.3*torch.ones(self.horizon, 1) # 0.1*torch.arange(1, self.horizon+1).unsqueeze(1)
        self.cov = torch.stack((vel_cov, steer_cov)).transpose(2, 0)
        # Define parameter to adjust for high weight updates
        self.gamma = gamma
        self.beta = 0.5

    def sample_new(self, batches=1):
        cov = self.cov.repeat(batches, 1, 1)
        mu = torch.zeros(batches, self.horizon, 2)
        noise = torch.normal(mean=mu, std=cov)
        sequence = []
        for i in range(0, self.horizon):
            if i == 0:
                # continue
                temp = self.beta * (self.mean[:, i+1] + noise[:, i, :]) + (1-self.beta) * self.mean[:, i]
            elif i == (self.horizon-1):
                temp = self.beta * (self.mean[:, -1] + noise[:, i, :]) + (1 - self.beta) * sequence[-1]
            else:
                temp = self.beta * (self.mean[:, i+1] + noise[:, i, :]) + (1-self.beta) * sequence[-1]
            sequence.append(temp)
        sequence = torch.stack(sequence, dim=1)

        # Clamp velocity between [0.1, 1.59] m/s
        sequence[:, :, 0] = torch.where(sequence[:, :, 0] < 0.1, 0.1, sequence[:, :, 0])
        sequence[:, :, 0] = torch.where(sequence[:, :, 0] > 1.59, 1.59, sequence[:, :, 0])
        # Clamp angle between [-0.95, 0.95]
        sequence[:, :, 1] = torch.where(sequence[:, :, 1] < -0.95, -0.95, sequence[:, :, 1])
        sequence[:, :, 1] = torch.where(sequence[:, :, 1] > 0.95, 0.95, sequence[:, :, 1])

        sequence = sequence.float()
        return sequence

    def update_new(self, reward, sequence):
        # reward is a batch x horizon x 1 tensor, sequence is a batch x horizon x 2 tensor
        mean = torch.zeros(self.horizon, 2)
        s_R = torch.zeros(self.horizon, 1)
        for r, seq in zip(reward, sequence):
            mean += np.exp(self.gamma * r) * seq
            s_R += np.exp(self.gamma * r)
        self.mean = (mean / s_R).transpose(0, 1).double()


if __name__ == "__main__":
    test = BadgrPlan()
    samp = test.sample_new(batches=3)
    # samp1 = test.sample_new()
    # samp = torch.stack((samp, samp1), 0)
    print(samp)
    # R = torch.tensor(np.random.rand(3, 5))
    # samp = samp.unsqueeze(0)
    # test.update_new(R, samp)
    # print(test.mean)
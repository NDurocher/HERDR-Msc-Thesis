

import numpy as np
# from PIL import Image
# from torch import nn
import torch
# import cv2


class BadgrPlan:
    def __init__(self, Horizon=5, vel_init=1.5, steer_init=0, gamma=50):
        # Set default starting value for actions
        # [Velocity (m/s), Steering angle (rad)]
        self.horizon = Horizon
        vel_mean = vel_init*np.ones(self.horizon)
        steer_mean = steer_init*np.zeros(self.horizon)
        # self.mean = np.concatenate((vel_mean, steer_mean))
        self.mean = np.vstack((vel_mean, steer_mean))
        # set guess for variance
        vel_cov = 0.1*np.ones(self.horizon)
        steer_cov = 0.1*np.ones(self.horizon)
        self.cov = np.diag(np.concatenate((vel_cov, steer_cov)))
        # Define parameter to adjust for high weight updates
        self.gamma = gamma
        self.beta = 0.3

    def sample(self, batches):
        sequence = np.random.multivariate_normal(np.zeros(2*self.horizon), self.cov, batches).reshape(
            [batches, self.horizon*2], order='C')
        for i in range(0, self.horizon-1):
            if i == 0:
                sequence[:,  i] = self.beta * sequence[:,  i]
                sequence[:,  i+self.horizon] = self.beta * sequence[:,  i + self.horizon]
            else:
                sequence[:,  i] = self.beta * sequence[:,  i] + (1-self.beta)*sequence[:,  i-1]
                sequence[:,  i+self.horizon] = self.beta * sequence[:,  i + self.horizon] +\
                                                 (1 - self.beta) * sequence[:,  i - 1 + self.horizon]
        sequence = sequence + np.broadcast_to(self.mean, (batches,  2*self.horizon))

        # Clamp velocity between [0,3] m/s
        sequence[:, 0:self.horizon-1] = np.where(sequence[:,  0:self.horizon-1] < 0, 0,
                                                    sequence[:,  0:self.horizon-1])
        sequence[:, 0:self.horizon - 1] = np.where(sequence[:,  0:self.horizon - 1] > 3, 3,
                                                      sequence[:,  0:self.horizon - 1])
        # Clamp angle between [-0.95, 0.95]
        sequence[:, self.horizon:(2*self.horizon) - 1] = np.where(sequence[:, self.horizon:(2*self.horizon) - 1]
                                                     < -0.95, -0.95, sequence[:, self.horizon:(2*self.horizon) - 1])
        sequence[:, self.horizon:(2*self.horizon) - 1] = np.where(sequence[:, self.horizon:(2*self.horizon) - 1]
                                                     > 0.95, 0.95, sequence[:, self.horizon:(2*self.horizon) - 1])

        return torch.tensor(sequence, dtype=torch.float32)

    def sample_new(self, batches=None):
        sequence = np.random.multivariate_normal(np.zeros(2 * self.horizon), self.cov).reshape(
            [2, self.horizon], order='C')
        for i in range(0, self.horizon-1):
            if i == 0:
                # continue
                sequence[:,  i] = self.beta * sequence[:,  i]
            else:
                sequence[:,  i] = self.beta * sequence[:,  i] + (1-self.beta)*sequence[:,  i-1]
        sequence = sequence + self.mean

        # Clamp velocity between [0,3] m/s
        sequence[0, :] = np.where(sequence[0, :] < 0, 0, sequence[0, :])
        sequence[0, :] = np.where(sequence[0, :] > 3, 3, sequence[0, :])
        # Clamp angle between [-0.95, 0.95]
        sequence[1, :] = np.where(sequence[1, :] < -0.95, -0.95, sequence[1, :])
        sequence[1, :] = np.where(sequence[1, :] > 0.95, 0.95, sequence[1, :])

        sequence = torch.tensor(sequence, dtype=torch.float32)
        sequence = sequence.unsqueeze(0)

        if batches is not None:
            for i in range(batches-1):
                sequence = torch.cat((sequence, self.sample_new()), 0)

        return sequence

    def update(self, reward, sequence):
        # reward is a (batch x) 1 x horizon vector, sequence is a batch x 2 x horizon matrix
        # 1 x (horizon*2) = sum_num_samples( 1 x (horizon*2) )
        mean = 0
        s_R = 0
        for i in range(sequence.shape[2]):
            mean += np.exp(self.gamma * float(reward[i]))*sequence[:, i]
            s_R += np.exp(self.gamma*float(reward[i]))
        self.mean = mean/s_R

    def update_new(self, reward, sequence):
        # reward is a (batch x) 1 x horizon vector, sequence is a batch x 2 x horizon matrix
        mean = 0
        s_R = 0
        for r, seq in zip(reward, sequence):
            mean += np.exp(self.gamma * r) * seq
            s_R += np.exp(self.gamma * r)
        self.mean = (mean / s_R).numpy()


if __name__ == "__main__":
    test = BadgrPlan()
    samp = test.sample_new(batches=3)
    # samp1 = test.sample_new()
    # samp = torch.stack((samp, samp1), 0)
    # print(samp.shape, "\n")
    R = torch.tensor(np.random.rand(3, 5))
    # samp = samp.unsqueeze(0)
    test.update_new(R, samp)
    print(test.mean)
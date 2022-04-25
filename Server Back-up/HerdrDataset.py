from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torchvision
import os

class HerdrDataset(Dataset):
    """  dataset."""

    def __init__(self, data_list, im_dir, transform=None):
        """
        Args:
            truth (string): Path to the txt file ground truth rewards.
            actions (string): Path to the txt file of actions.
            images (string): Path to the images folder
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_pickle(data_list)
        self.im_dir = im_dir

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        action = torch.nan_to_num(self.df['Actions'].values[idx])
        gnd = torch.nan_to_num(self.df['Ground_Truth'].values[idx])
        img_name = os.path.join(self.im_dir,
                                str(self.df['Image_Name'].values[idx]))
        im = torchvision.io.read_image(img_name).float()
        if self.transform:
            im = self.transform(im)

        sample = [im, action, gnd]
        return sample


if __name__ == "__main__": 
    
    data_dir_train = "./Herdr_data_train.pkl"
    data_dir_test = "./Herdr_data_test.pkl"
    img_dir_train = "./images"
    img_dir_test = "./Test"

    # validation_size = 0.3

    train_data = HerdrDataset(data_dir_train, img_dir_train)
    test_data = HerdrDataset(data_dir_test, img_dir_test)

    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(validation_size * num_train))
    # np.random.shuffle(indices)

    from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler 
    # train_idx, test_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # test_sampler = SubsetRandomSampler(test_idx)

    train_sampler = RandomSampler(train_data)
    test_sampler = RandomSampler(test_data)

    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)  # sampler=test_sampler
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=32)  # sampler=test_sampler
    # del indices, train_idx, test_idx
    # del train_sampler, test_sampler
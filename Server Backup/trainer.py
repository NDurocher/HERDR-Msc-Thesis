from datetime import datetime
import numpy as np
import torch
import os
from torch import nn
from torchvision.transforms.functional import resize
from torchvision.transforms import Resize
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler 
from Badgrnet import HERDR
from HDF5DataClass_new import HDF5Dataset

def one_epoch(model, data_loader, epoch, opt=None):
    train = False if opt is None else True
    model.train() if train else model.eval()
    losses, correct, total = [], [], 0
    incorrect = 0
    criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor(8.05).to(device))
    sig = nn.Sigmoid()
    step = 0 + epoch*len(data_loader)
    for x, y, z in data_loader:
        model.zero_grad()
        x, y, z = x.to(device), y.to(device), z.to(device) 
        with torch.set_grad_enabled(train):
            logits = model(x,y)
        loss = criterion(logits, z.unsqueeze(2))
        
        samples = z.shape[0]*z.shape[1]
        total += samples
        pos_samples = torch.count_nonzero(z)
        pos_correct = torch.count_nonzero(torch.logical_and( (abs(z-sig(logits[:,:,0])) < 0.50), z)).item()
        correct.append(torch.count_nonzero(abs(z-sig(logits[:,:,0])) < 0.50).item())
        incorrect = torch.count_nonzero(abs(sig(logits[:,:,0])-z) >= 0.50).item()
        
        if step % 200 == 0:
            rn_idxs = torch.randint(0,z.shape[0],(2,))
            print(f"Sample output:\n{sig(logits)[rn_idxs].squeeze(2).detach().cpu()}")
            print(f"Sample GND:\n{z[rn_idxs].cpu()}")
        del logits, x, y, z
        
        if train:
            loss.backward()
            opt.step()
            opt.zero_grad()


        losses.append(loss.item())
        
        writer.add_scalar("Loss/valid", losses[-1], step) if opt is None else writer.add_scalar("Loss/train", losses[-1], step)
        writer.add_scalar("Accuracy/valid", correct[-1]/samples, step) if opt is None else writer.add_scalar("Accuracy/train", correct[-1]/samples, step)
        writer.add_scalar("Incorrect/valid", incorrect/samples, step) if opt is None else writer.add_scalar("Incorrect/train", incorrect/samples, step)
        if pos_samples > 0:
            writer.add_scalar("Pos_Accuracy/valid", pos_correct/pos_samples, step) if opt is None else writer.add_scalar("Pos_Accuracy/train", pos_correct/pos_samples, step)
        del loss
        
        step += 1
        
    return np.mean(losses), sum(correct)/ total


def train(model, loader_train, loader_valid, time, dir_path, lr=1e-4, max_epochs=30, weight_decay=1e-1, patience=2):
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid_accuracy = 0
    best_valid_loss = 10000
    best_valid_acc_epoch = 0
    t = tqdm(range(max_epochs))
    # for param in model.parameters():
    #         param.requires_grad = False
    # for param in model.obs_pre.parameters():
    #         param.requires_grad = True
    # model = model.cuda()

    for epoch in t:
        train_loss, train_acc = one_epoch(model, loader_train, epoch, opt)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        valid_loss, valid_acc = one_epoch(model, loader_valid, epoch)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        
        t.set_description(f'train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}')
        print(f"Train Accuracy: {train_acc:.4f} \nValid Accuracy: {valid_acc:.4f}\n")
        

        if valid_acc > best_valid_accuracy:
            best_valid_accuracy = valid_acc
            best_valid_acc_epoch = epoch
            torch.save(model, f'{dir_path}/models/Herdr_cross'+time+'.pth')

        if epoch > best_valid_acc_epoch + patience:
            print(f"Best model epoch: {best_valid_acc_epoch}")
            break
    t.set_description(f'best valid loss: {best_valid_loss:.2f}')
    print(f" Train Accuracy: {train_acc}")
    print(f" Validation Accuracy: {valid_acc}")
    # writer.flush()

    return train_losses, train_accuracies, valid_losses, valid_accuracies



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Use GPU")
    else:
        device = torch.device('cpu')
        print("Use CPU")
    dir_path = os.path.abspath(os.getcwd())

    data_dir_train = f"{dir_path}/hdf5s"
    # img_dir_train = f"{dir_path}/images"
    data_dir_test = f"{dir_path}/hdf5s_test"
    # img_dir_test = "/content/Test"

    batch_size = 50
    HRZ = 10
    loader_params = {'batch_size': batch_size, 'shuffle': True,'num_workers': 3}
    train_data = HDF5Dataset(data_dir_train, recursive=False, load_data=True,
                          data_cache_size=10000, transform=None)
    test_data = HDF5Dataset(data_dir_test, recursive=False, load_data=True,
                          data_cache_size=10000, transform=None)
    
    '''
    validation_size = 0.3
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(validation_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    '''
    # train_sampler = RandomSampler(train_data)
    # test_sampler = RandomSampler(test_data)

    trainloader = torch.utils.data.DataLoader(train_data, **loader_params)  # sampler=test_sampler
    testloader = torch.utils.data.DataLoader(test_data, **loader_params)

    print(f'Training epoch will take {len(trainloader)} steps')
    print(f'Validation epoch will take {len(testloader)} steps')

    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=2)
    model = HERDR(Horizon=HRZ, RnnDim=64)
    model.to(device)
    time = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")
    writer = SummaryWriter(log_dir=f'{dir_path}/logs/'+time)
    train_losses, train_accuracies, valid_losses, valid_accuracies = train(model, trainloader, testloader, time, dir_path)
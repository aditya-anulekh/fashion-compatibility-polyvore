import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp
import pickle as pkl

from utils import Config
from model import DualResNet
from data import get_pair_dataloader


def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):
    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    track_loss = {"train":[],
            "test":[]}
    
    track_acc = {"train":[],
            "test":[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for (image1, image2), labels in tqdm(dataloader[phase]):
                image1 = image1.to(device)
                image2 = image2.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(image1, image2)
                    outputs = outputs.reshape(image1.size(0))
                    pred = torch.Tensor([1 if x>=0.5 else 0 for x in outputs])
                    pred = pred.to(device)
                    loss = criterion(outputs, labels.float())

                    if phase=='train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * image1.size(0)
                running_corrects += torch.sum(pred==labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            track_loss[phase].append(epoch_loss)
            track_acc[phase].append(epoch_acc.to("cpu"))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase=='test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth'))
        print('Model saved at: {}'.format(osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth')))

    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))

    return track_loss, track_acc


if __name__=='__main__':

    dataloaders, dataset_size = get_pair_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])
    model = DualResNet()
    model.model1.requires_grad = False
    # model.model2.requires_grad = False

    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=Config['learning_rate'])
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
    print(device)

    loss, acc = train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size)

    fig, ax = plt.subplots(1, 2)
    fig.axes[0].plot(loss["train"])
    fig.axes[0].plot(loss["test"])

    fig.axes[1].plot(acc["train"])
    fig.axes[1].plot(acc["test"])
    fig.savefig("plots.png", dpi=200)
    
    # with open("variables.pkl", "wb") as file:
    #     pkl.dump([le, loss, acc], file)

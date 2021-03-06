import os
import warnings
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
from vgg import DualVggNet
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

            for i, ((image1, image2), labels) in enumerate(tqdm(dataloader[phase])):
                image1 = image1.to(device)
                image2 = image2.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(image1, image2)
                    # outputs = outputs.reshape(image1.size(0))
                    _, pred = torch.max(outputs, 1)
                    pred = pred.to(device)
                    loss = criterion(outputs, labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * image1.size(0)
                running_corrects += torch.sum(pred==labels.data)

                if i%500 == 0:
                    tqdm.write(f"{i} | loss - {loss.item()} | acc - {torch.sum(pred==labels.data)}")

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            track_loss[phase].append(epoch_loss)
            track_acc[phase].append(epoch_acc.to("cpu"))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase=='test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        checkpoints_dir = "checkpoints_031221"

        if not Config["debug"]:

            torch.save(best_model_wts, osp.join(checkpoints_dir, f'model_{epoch}.pth'))
            print('Model saved at: {}'.format(osp.join(checkpoints_dir, f'model_{epoch}.pth')))

            with open(f"{checkpoints_dir}/variables.pkl", "wb") as file:
                pkl.dump([track_loss, track_acc], file)


    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))

    return track_loss, track_acc


if __name__=='__main__':
    if Config["debug"]:
        warnings.warn("DEBUG is set to True. Set to False when running actual training!!")

    dataloaders, dataset_size = get_pair_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])
    model = DualVggNet()
    print(model)
    # model.model1.requires_grad = False
    # model.model2.requires_grad = False

    # model.load_state_dict(torch.load("checkpoints_021221/model_7.pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=Config['learning_rate'])
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
    print(device)
    print(os.getcwd())
    if not osp.exists(f"checkpoints"):
        os.mkdir(f"checkpoints")

    loss, acc = train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size)

    fig, ax = plt.subplots(1, 2)
    fig.axes[0].plot(loss["train"], label="train")
    fig.axes[0].plot(loss["test"], label="validation")
    fig.axes[0].set_title("Training and Validation Loss")
    fig.axes[0].legend()

    fig.axes[1].plot(acc["train"], label="train")
    fig.axes[1].plot(acc["test"], label="validation")
    fig.axes[1].set_title("Training and Validation Accuracy")
    fig.axes[1].legend()

    fig.savefig("plots.png", dpi=200)
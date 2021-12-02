import os
import os.path as osp
import json
import pickle as pkl
from tqdm import tqdm

import torch
from torchvision import transforms, io
from torch.utils.data import Dataset, DataLoader, dataloader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from utils import Config
from PIL import Image
import numpy as np
from model import model


class PolyvoreImageLoader:
    def __init__(self):
        self.root_dir = Config["root_path"]
        self.image_dir = osp.join(self.root_dir, "images")
        self.transforms = self.get_data_transforms()

    def get_data_transforms(self):
        data_transforms = {
            "train":transforms.Compose([
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize([0.5 for _ in range(Config["channels"])],
                [0.5 for _ in range(Config["channels"])])
            ]),
            "test":transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize([0.5 for _ in range(Config["channels"])],
                [0.5 for _ in range(Config["channels"])])
            ])
        }

        return data_transforms

    def create_dataset(self):
        with open(Config["meta_file"], "r") as file:
            meta_json = json.load(file)

        categories = pd.read_csv(Config["category_file_path"], names=["id", "fine", "semantic"])

        id_to_category = {}
        for k, v in tqdm(meta_json.items()):
            id_to_category[k] = v["category_id"]

        files = os.listdir(self.image_dir)
        X = []
        y = []

        for image in tqdm(files):
            if id_to_category.get(osp.splitext(image)[0], False):
                X.append(image)
                category_id = int(id_to_category[osp.splitext(image)[0]])
                # category = categories[categories["id"]==category_id]["semantic"].values[0]
                y.append(category_id)


        # Encoding y
        le = LabelEncoder()
        le.fit(y)
        y = le.fit_transform(y)
        print(f"# of images: {len(X)} \n# of categories: {max(y)+1}")

        # Split the dataset for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        return X_train, X_test, y_train, y_test, max(y)+1, le


class PolyvoreDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform
        self.image_dir = osp.join(Config["root_path"], "images")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.to_list()
        
        img_name = osp.join(self.image_dir, self.X[item])
        img = io.read_image(img_name).float()
        if self.transform:
            img = self.transform(img)

        return img, self.y[item]


def get_dataloader(debug, batch_size, num_workers):
    dataset = PolyvoreImageLoader()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, classes, le = dataset.create_dataset()

    if debug:
        debug_set_size = 200
        train_set = PolyvoreDataset(X_train[:debug_set_size], y_train[:debug_set_size], transform=transforms['train'])
        test_set = PolyvoreDataset(X_test[debug_set_size:2*debug_set_size], 
                                    y_test[debug_set_size:2*debug_set_size], 
                                    transform=transforms['test'])
        dataset_size = {'train': debug_set_size, 'test': debug_set_size}
    else:
        train_set = PolyvoreDataset(X_train, y_train, transforms['train'])
        test_set = PolyvoreDataset(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    datasets = {'train': train_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'test']}
    return dataloaders, classes, dataset_size, le



# For pairwise classification

class PolyvorePairDataset(Dataset):
    def __init__(self, X, y, transform, debug=False):
        self.X = X
        self.y = y
        if debug:
            debug_length = 100
            self.X = self.X[len(self.X)//2-debug_length:len(self.X)//2+debug_length]
        self.image_dir = osp.join(Config["root_path"], "images")
        self.transform = transform

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, index):
        images = []
        y = self.y[index]
        for image in self.X[index]:
            img_name = osp.join(self.image_dir, image)
            images.append(self.transform(io.read_image(img_name).float()))

        return images, y


def get_pair_dataloader(debug, batch_size, num_workers):
    data_transforms = {
            "train":transforms.Compose([
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize([0.5 for _ in range(Config["channels"])],
                [0.5 for _ in range(Config["channels"])])
            ]),
            "test":transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize([0.5 for _ in range(Config["channels"])],
                [0.5 for _ in range(Config["channels"])])
            ])
        }

    # Load train and valid files
    with open("train_files/X_train.pkl", "rb") as file:
        X_train = pkl.load(file)

    with open("train_files/y_train.pkl", "rb") as file:
        y_train = pkl.load(file)

    with open("train_files/X_valid.pkl", "rb") as file:
        X_valid = pkl.load(file)

    with open("train_files/y_valid.pkl", "rb") as file:
        y_valid = pkl.load(file)


    train_dataset = PolyvorePairDataset(X_train, y_train,
                                        data_transforms["train"], debug)
    test_dataset = PolyvorePairDataset(X_valid, y_valid,
                                        data_transforms["test"], debug)

    dataset_size = {'train': train_dataset.__len__(), 
                    'test': test_dataset.__len__()}

    datasets = {'train': train_dataset, 'test': test_dataset}

    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'test']}
    return dataloaders, dataset_size


if __name__ == "__main__":
    dataloaders, dataset_size = get_pair_dataloader(False, 64, 1)
    model.fc = torch.nn.Identity()
    # model.to("cuda")
    for (image1, image2), y in dataloaders["train"]:
        image1 = image1
        image2 = image2
        print(image1.size(0))
        break
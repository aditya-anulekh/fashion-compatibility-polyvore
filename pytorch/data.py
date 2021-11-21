import os
import os.path as osp
import json
from tqdm import tqdm

import torch
from torchvision import transforms, io
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import Config


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
        id_to_category = {}
        for k, v in tqdm(meta_json.items()):
            id_to_category[k] = v["category_id"]

        files = os.listdir(self.image_dir)
        X = []
        y = []

        for image in files:
            if id_to_category.get(osp.splitext(image)[0], False):
                X.append(image)
                y.append(int(id_to_category[osp.splitext(image)[0]]))

        # Encoding y
        y = LabelEncoder().fit_transform(y)
        print(f"# of images: {len(X)} \n# of categories: {max(y)+1}")

        # Split the dataset for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        return X_train, X_test, y_train, y_test, max(y)+1


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
    X_train, X_test, y_train, y_test, classes = dataset.create_dataset()

    if debug:
        train_set = PolyvoreDataset(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = PolyvoreDataset(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
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
    return dataloaders, classes, dataset_size

if __name__ == "__main__":
    print(Config["debug"])
    print(get_dataloader(Config["debug"], 64, 1))
import random
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Define a training image loader that specifies transforms on images
train_transformer = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Transform it into a torch tensor
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize
])

mask_transformer = transforms.Compose([
    transforms.Resize((224, 224))
])

# Define a evaluation image loader that specifies transforms on images
eval_transformer = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Transform it into a torch tensor
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize
])

class GetDataset(Dataset):
    def __init__(self, data_dir, mask_dir, dataset_type, num_classes, transform, mask_transform):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.dataset_type = dataset_type
        self.dataset = []
        self.num_classes = num_classes
        with open(self.dataset_type, 'r') as f:
            for line in f.readlines():
                self.dataset.append('{}.png'.format(line.strip()))
        self.mask_filenames = os.listdir(mask_dir)
        self.mask_filenames = [os.path.join(mask_dir, f) for f in self.dataset]
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        mask = Image.open(self.mask_filenames[idx])
        image = Image.open(self.mask_filenames[idx].replace('png', 'jpg').replace(self.mask_dir, self.data_dir))
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = np.array(mask)
        mask = np.where(mask==255,0,mask)
        mask_new = np.zeros([self.num_classes, mask.shape[0], mask.shape[1]])
        for idx in range(mask.shape[0]):
            for idy in range(mask.shape[1]):
                mask_new[mask[idx, idy], idx, idy] = 1
        mask = torch.from_numpy(mask_new)
        return image, mask

def fetch_dataloader(types, data_dir, mask_dir, dataset_dir, num_classes, params):
    dataloaders={}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir)
            mask_path = os.path.join(mask_dir)
            dataset_type = os.path.join(dataset_dir, "{}.txt".format(split))

            if split == 'train':
                dl = DataLoader(GetDataset(path, mask_path, dataset_type, num_classes, train_transformer, mask_transformer), batch_size=params.batch_size, shuffle=True,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)
            else:
                dl = DataLoader(GetDataset(path, mask_path, dataset_type, num_classes, eval_transformer, mask_transformer), batch_size=params.batch_size, shuffle=True,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)
            dataloaders[split] = dl
    return dataloaders
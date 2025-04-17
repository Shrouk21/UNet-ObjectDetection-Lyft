import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl


class DataProcessing(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.mask_names = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.bmp'))])
        assert len(self.image_names) == len(self.mask_names), "Mismatched number of images and masks"
        #map pixel values to class indices in mask
        self.mapping = self.dynamic_mapping(mask_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.vectorize(mapping.get)(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = torch,tensor(augmented['mask'], dtype=torch.long)
        return image, mask
    def dynamic_mapping(self, mask_dir):
        unique_values = set()
        for mask_name in os.listdir(mask_dir):
            mask_path = os.path.join(mask_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            unique_values.update(np.unique(mask))
            
        return {value:idx for idx, value in enumerate(unique_values))
    
        # # Get unique pixel values and sort them
        # unique_values = np.unique(mask)
        # mapping = {value:idx for idx, value in enumerate(unique_values)}

        # #apply mapping to the mask
        # mapped_mask = np.vectorize(mapping.get)(mask)
        # return mapped_mask, mapping

class Dataloader(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, val_split=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def setup(self, stage=None):
        train_image_dir = os.path.join(self.data_dir, "train_val", "images")
        train_mask_dir = os.path.join(self.data_dir, "train_val", "masks")
    
    
        print("Looking for training images in:", train_image_dir)
        print("Looking for training masks in:", train_mask_dir)
    
        full_dataset = DataProcessing(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=self.transform)
        val_size = int(self.val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])


        
        test_image_dir = os.path.join(self.data_dir, "TEST", "images")
        test_mask_dir = os.path.join(self.data_dir, "TEST", "masks")
        self.test_dataset = DataProcessing(image_dir=test_image_dir, mask_dir=test_mask_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

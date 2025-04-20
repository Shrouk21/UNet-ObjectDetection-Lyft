import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from PIL import Image


class DataProcessing(Dataset):
    def __init__(self, class_map,  image_dir, mask_dir, transform=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.class_map = class_map
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.mask_names = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', 'png'))])
        assert len(self.image_names) == len(self.mask_names), "Mismatched number of images and masks"
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
    
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
    
        image = Image.open(img_path).convert("RGB") 
        mask = Image.open(mask_path).convert("RGB") 
        mask = np.array(mask)  
        h, w, _ = mask.shape
        mask = mask.reshape(-1, 3)
        
        # Map RGB values to class indices
        class_indices = np.array([self.class_map.get(tuple(rgb), -1) for rgb in mask])
        
        mask = class_indices.reshape(h, w)  # Reshape back to [H, W]
        if np.any(mask==-1):
            mask[mask == -1] = 0
    
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=mask)
            image = augmented['image']  
            mask = augmented['mask']    

      
        mask = mask.to(torch.long)

        image, mask = image.to(self.DEVICE), mask.to(self.DEVICE)
    
        return image, mask

class Dataloader(pl.LightningDataModule):
    def __init__(self, data_dir, class_map, batch_size=32, num_workers=4, val_split=0.2, test_split=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.class_map = class_map

    def setup(self,stage=None):
        img_subpath, mask_subpath = os.listdir(self.data_dir)
        train_image_dir = os.path.join(self.data_dir, img_subpath)
        train_mask_dir = os.path.join(self.data_dir, mask_subpath)
    
    
        print("Looking for training images in:", train_image_dir)
        print("Looking for training masks in:", train_mask_dir)
    
        full_dataset = DataProcessing(class_map=self.class_map, image_dir=train_image_dir, mask_dir=train_mask_dir)
        dataset_size = len(full_dataset)
        val_size = int(self.val_split * dataset_size)
        test_size = int(self.test_split * dataset_size)
        train_size = dataset_size - val_size - test_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size])


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

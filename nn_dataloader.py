import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import cv2
import numpy as np
from PIL import Image
import glob
import os
from tqdm import tqdm
from utils import compute_bbox_ellipse, boxes_to_yoloLabel

class FDDBDataset(Dataset):
    """
    create a Dataset from a single folder
    """
    def __init__(self, label_dir, img_dir, img_size=448, S=7, B=2, C=0):
        self.imgs = []
        self.labels = []
        self.S = S
        self.B = B
        self.C = C
        self.img_size = img_size
        with open(label_dir) as f:
            img_path = f.readline()[:-1]
            while img_path != "":
                img = np.asarray(Image.open(os.path.join(img_dir, img_path + ".jpg")))
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                self.imgs.append(img)
                num_boxes = int(f.readline()[:-1])
                boxes = np.zeros([num_boxes, 4])
                for i in range(num_boxes):
                    major_axis_radius, minor_axis_radius, angle, center_x, center_y, valid = f.readline()[:-1].split()
                    assert(valid == '1')
                    max_x, min_x, max_y, min_y = compute_bbox_ellipse(float(major_axis_radius), float(minor_axis_radius), float(angle), float(center_x), float(center_y))
                    max_x, min_x, max_y, min_y = min(img.shape[1]-1, max_x), max(0, min_x), min(img.shape[0]-1, max_y), max(0, min_y)
                    boxes[i, :] = (min_x + max_x) / 2, (min_y + max_y) / 2, max_x - min_x, max_y - min_y
                self.labels.append(boxes.astype(int))
                img_path = f.readline()[:-1]
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        label = boxes_to_yoloLabel(self.imgs[idx].shape[0], self.imgs[idx].shape[1], self.labels[idx], self.S, self.B, self.C)
        # img = torch.as_tensor(self.imgs[idx])
        img = torch.as_tensor(cv2.resize(self.imgs[idx], (self.img_size, self.img_size))).unsqueeze(0)
        return img, label

class FDDBDataloader(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=8, num_workers=4, img_size=448, S=7, B=2, C=0):
        super(FDDBDataloader, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        label_paths = glob.glob(os.path.join(data_path, "FDDB-folds/*[1-8]-ellipseList.txt"))
        img_dir = os.path.join(data_path, "originalPics/")
        self.train_dataset, self.val_dataset = [], []

        for i, label_path in enumerate(tqdm(label_paths)):
            print('loading fold: ', label_path)
            dataset = FDDBDataset(label_path, img_dir, img_size, S, B, C)
            if i < int(len(label_paths) * 0.75):
                self.train_dataset.append(dataset)
            else:
                self.val_dataset.append(dataset)
        
        self.train_dataset, self.val_dataset = ConcatDataset(self.train_dataset), ConcatDataset(self.val_dataset)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                drop_last=True if len(self.train_dataset) % self.batch_size != 0 else False, 
                pin_memory=True,
                shuffle=True
                )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                drop_last=True if len(self.val_dataset) % self.batch_size != 0 else False, 
                pin_memory=True,
                shuffle=False
                )

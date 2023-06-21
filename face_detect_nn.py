import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from dataloader import dataloader, dataset_for_classification
import tensorboard

class FaceClassifier(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _, y_pred = torch.max(y_hat, dim=1)
        accuracy = torch.sum(y_pred == y).item() / (len(y) * 1.0)
        self.log("test_accuracy", accuracy)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size=32, num_workers=2):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.trainset = MyDataset(self.data, transform=None, train=True)
        self.testset = MyDataset(self.data, transform=None, train=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, train=True):
        self.transform = transform
        self.train = train
        self.data = data
    
    def __len__(self):
        return len(self.data['img'])

    def __getitem__(self, idx):
        image = torch.from_numpy(np.float32(self.data['img'][idx][np.newaxis, :]))
        label = self.data['labels'][idx]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    train_data, test_data = dataloader("./FDDB-folds/", "./originalPics/")
    train_classification = dataset_for_classification(train_data, 10)
    test_classification = dataset_for_classification(test_data, 10)
    model = FaceClassifier()
    dm = MyDataModule(train_classification)
    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(model, dm)
    test_dm = MyDataModule(test_classification)
    trainer.test(model, datamodule=test_dm)
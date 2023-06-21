from typing import Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils import iou, yoloLabel_to_boxes
import numpy as np

architecture_config = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: tuples and then last integer represents number of repeats
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(pl.LightningModule):
    def __init__(self, in_channels, split_size, num_boxes, num_classes, lr=3e-4, weight_decay=1e-5, lambda_noobj=0.05, lambda_coord=5):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, self.S * self.S * (self.C + self.B * 5))
        )
        # label: [c1,c2,...,c20,p_c,x,y,w,h...(repeat last 5 places B times)] where c's are classes and p_c is the probability that there is an object
        self.mse = nn.MSELoss(reduction="mean")
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord

    def forward(self, x):
        x = self.darknet(x)
        return self.fc(torch.flatten(x, start_dim=1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x).reshape(-1, self.S, self.S, self.C + self.B*5)
        iou_b1 = iou(y_hat[..., self.C+1:self.C+5], y[..., self.C+1:self.C+5])
        iou_b2 = iou(y_hat[..., self.C+6:self.C+10], y[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        box_exists = y[..., self.C].unsqueeze(3)

        # BOX COORDINATE LOSS
        box_predictions = box_exists * (
            # check which bbox is responsible (has a high probability)
            bestbox * y_hat[..., self.C+6:self.C+10] + 
            (1 - bestbox) * y_hat[..., self.C+1:self.C+5]
        )
        box_targets = box_exists * y[..., self.C+1:self.C+5]
        # take the square root of the width and the height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4].clone()) * \
                torch.sqrt(torch.abs(box_predictions[..., 2:4].clone()) + 1e-6)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4].clone())
        box_loss = self.mse(
            # (N, S, S, 4) -> (N*S*S, 4)
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # OBJECT LOSS
        pred_box = (
            bestbox * y_hat[..., self.C+5:self.C+6] + (1 - bestbox) * y_hat[..., self.C:self.C+1]
        )
        object_loss = self.mse(
            # (N, S, S, 1) -> (N*S*S)
            torch.flatten(box_exists * pred_box),
            torch.flatten(box_exists * y[..., self.C:self.C+1])
        )

        # NO OBJECT LOSS (Both boxes)
        no_object_loss = self.mse(
            # (N, S, S, 1) -> (N*S*S)
            torch.flatten((1 - box_exists) * y_hat[..., self.C:self.C+1]),
            torch.flatten((1 - box_exists) * y[..., self.C:self.C+1])
        )
        no_object_loss += self.mse(
            # (N, S, S, 1) -> (N*S*S)
            torch.flatten((1 - box_exists) * y_hat[..., self.C+5:self.C+6]),
            torch.flatten((1 - box_exists) * y[..., self.C:self.C+1])
        )

        # CLASS LOSS (UNNECESSARY For 1 class)

        loss = self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss
        self.log('train_box_loss', box_loss, prog_bar=True, logger=True, on_epoch=True)
        self.log('train_object_loss', object_loss, prog_bar=True, logger=True, on_epoch=True)
        self.log('train_no_object_loss', no_object_loss, prog_bar=True, logger=True, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x).reshape(-1, self.S, self.S, self.C + self.B*5)
        iou_b1 = iou(y_hat[..., self.C+1:self.C+5], y[..., self.C+1:self.C+5])
        iou_b2 = iou(y_hat[..., self.C+6:self.C+10], y[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        box_exists = y[..., self.C].unsqueeze(3)

        # BOX COORDINATE LOSS
        box_predictions = box_exists * (
            # check which bbox is responsible (has a high probability)
            bestbox * y_hat[..., self.C+6:self.C+10] + 
            (1 - bestbox) * y_hat[..., self.C+1:self.C+5]
        )
        box_targets = box_exists * y[..., self.C+1:self.C+5]
        # take the square root of the width and the height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4].clone()) * \
                torch.sqrt(torch.abs(box_predictions[..., 2:4].clone()) + 1e-6)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4].clone())
        box_loss = self.mse(
            # (N, S, S, 4) -> (N*S*S, 4)
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # OBJECT LOSS
        pred_box = (
            bestbox * y_hat[..., self.C+5:self.C+6] + (1 - bestbox) * y_hat[..., self.C:self.C+1]
        )
        object_loss = self.mse(
            # (N, S, S, 1) -> (N*S*S)
            torch.flatten(box_exists * pred_box),
            torch.flatten(box_exists * y[..., self.C:self.C+1])
        )

        # NO OBJECT LOSS (Both boxes)
        no_object_loss = self.mse(
            # (N, S, S, 1) -> (N*S*S)
            torch.flatten((1 - box_exists) * y_hat[..., self.C:self.C+1]),
            torch.flatten((1 - box_exists) * y[..., self.C:self.C+1])
        )
        no_object_loss += self.mse(
            # (N, S, S, 1) -> (N*S*S)
            torch.flatten((1 - box_exists) * y_hat[..., self.C+5:self.C+6]),
            torch.flatten((1 - box_exists) * y[..., self.C:self.C+1])
        )

        # CLASS LOSS (UNNECESSARY For 1 class)

        loss = self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss
        self.log('val_box_loss', box_loss, prog_bar=True, logger=True, on_epoch=True)
        self.log('val_object_loss', object_loss, prog_bar=True, logger=True, on_epoch=True)
        self.log('val_no_object_loss', no_object_loss, prog_bar=True, logger=True, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if batch_idx < 10:
            with torch.no_grad():
                img, gt_box = batch
                pred_box = self.forward(img.float()).reshape(-1, self.S, self.S, self.C + self.B*5)
                # convert grayscale image to 3 channel
                img = img[0][0].detach().cpu().numpy()
                img = np.stack((img, img, img), axis=-1)
                gt_box = gt_box[0].detach().cpu().numpy()
                pred_box = pred_box[0].detach().cpu().numpy()
                gt_box = yoloLabel_to_boxes(img.shape[0], img.shape[1], gt_box, self.S, self.B, self.C)
                pred_box = yoloLabel_to_boxes(img.shape[0], img.shape[1], pred_box, self.S, self.B, self.C)
                for _, cx, cy, w, h in gt_box:
                    x1 = int(min(max(cx - w/2, 0), img.shape[1]-1))
                    x2 = int(min(max(cx + w/2, 0), img.shape[1]-1))
                    y1 = int(min(max(cy - h/2, 0), img.shape[0]-1))
                    y2 = int(min(max(cy + h/2, 0), img.shape[0]-1))
                    img[y1:y2, x1, :] = [0, 255, 0]
                    img[y1:y2, x2, :] = [0, 255, 0]
                    img[y1, x1:x2, :] = [0, 255, 0]
                    img[y2, x1:x2, :] = [0, 255, 0]
                for _, cx, cy, w, h in pred_box:
                    x1 = int(min(max(cx - w/2, 0), img.shape[1]-1))
                    x2 = int(min(max(cx + w/2, 0), img.shape[1]-1))
                    y1 = int(min(max(cy - h/2, 0), img.shape[0]-1))
                    y2 = int(min(max(cy + h/2, 0), img.shape[0]-1))
                    img[y1:y2, x1, :] = [255, 0, 0]
                    img[y1:y2, x2, :] = [255, 0, 0]
                    img[y1, x1:x2, :] = [255, 0, 0]
                    img[y2, x1:x2, :] = [255, 0, 0]
                self.logger.experiment.add_image("img_{}".format(batch_idx), np.moveaxis(img, 2, 0), global_step=self.current_epoch)


    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                    in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]
                in_channels = x[1]
                
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)


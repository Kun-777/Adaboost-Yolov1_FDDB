from nn_dataloader import FDDBDataset
from torch.utils.data import ConcatDataset
from nn_model import Yolov1
import torch
from utils import yoloLabel_to_boxes, nonMaximumSuppression
import matplotlib.pyplot as plt
import numpy as np

def eval(ds, model, S, C, B, thres_iou):
    model.eval().to('cuda')
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for i in range(len(ds)):
            '''generate progress bar to show the progress of the evaluation every 10%'''
            if i % (len(ds) // 10) == 0:
                print("Progress: {}%".format(i // (len(ds) // 10) * 10))
            img, gt_label = ds[i]
            pred_label = model(img.to('cuda').unsqueeze(0).float()).reshape(-1, S, S, C + B*5)
            pred_label = pred_label[0].detach().cpu().numpy()
            pred_boxes = yoloLabel_to_boxes(img.shape[0], img.shape[1], pred_label, S, B, C)
            gt_boxes = yoloLabel_to_boxes(img.shape[0], img.shape[1], gt_label, S, B, C)
            if len(pred_boxes) > 0:
                pred_boxes = nonMaximumSuppression(pred_boxes, 0.5)
            # count tp, fp, fn
            num_correct = 0
            for bbox in pred_boxes:
                # calculate if this bbox is a true positive
                _, x, y, w, h = bbox
                min_x1 = x - 0.5 * w
                min_y1 = y - 0.5 * h
                max_x1 = min_x1 + w
                max_y1 = min_y1 + h
                correct = False
                for gt_bbox in gt_boxes:
                    _, x, y, w, h = gt_bbox
                    min_x2 = x - 0.5 * w
                    min_y2 = y - 0.5 * h
                    max_x2 = min_x2 + w
                    max_y2 = min_y2 + h
                    # find the intersection box
                    max_xi, min_xi, max_yi, min_yi = min(max_x1, max_x2), max(min_x1, min_x2), min(max_y1, max_y2), max(min_y1, min_y2)
                    wi = max_xi - min_xi
                    hi = max_yi - min_yi
                    if wi <= 0 or hi <= 0:
                        continue
                    # calculate intersection area
                    inter = wi * hi
                    union = (max_x1 - min_x1) * (max_y1 - min_y1) + (max_x2 - min_x2) * (max_y2 - min_y2) - inter
                    IoU = inter / union
                    if IoU > thres_iou:
                        correct = True
                        break
                if correct:
                    # this is a true positive
                    tp += 1
                    num_correct += 1
                else:
                    fp += 1
            fn += len(gt_boxes) - num_correct
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

if __name__ == "__main__":
    model = Yolov1(1,7,2,0)
    model.load_state_dict(torch.load("./model/new/epoch=43-val_loss=0.01518.ckpt")['state_dict'])
    eval_ds = ConcatDataset(
                [FDDBDataset("./FDDB-folds/FDDB-fold-09-ellipseList.txt", "./originalPics/"), 
                 FDDBDataset("./FDDB-folds/FDDB-fold-10-ellipseList.txt", "./originalPics/")]
                )
    precisions = []
    recalls = []
    for thres in [0.4,0.45,0.5,0.53,0.56,0.58,0.60]:
        precision, recall = eval(eval_ds, model, 7, 0, 2, thres)
        precisions.append(precision)
        recalls.append(recall)
    print(precisions, recalls)
    plt.plot(recalls, precisions)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.show()
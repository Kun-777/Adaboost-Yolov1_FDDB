import torch
import numpy as np

# bboxes are (x,y,w,h) where x, y are the coordinates of the center
def iou(bbox1, bbox2):
    min_x1 = bbox1[..., 0:1] - 0.5 * bbox1[..., 2:3]
    min_y1 = bbox1[..., 1:2] - 0.5 * bbox1[..., 3:4]
    max_x1 = min_x1 + bbox1[..., 2:3]
    max_y1 = min_y1 + bbox1[..., 3:4]
    min_x2 = bbox2[..., 0:1] - 0.5 * bbox2[..., 2:3]
    min_y2 = bbox2[..., 1:2] - 0.5 * bbox2[..., 3:4]
    max_x2 = min_x2 + bbox2[..., 2:3]
    max_y2 = min_y2 + bbox2[..., 3:4]
    # find the intersection box
    max_xi = torch.min(max_x1, max_x2)
    min_xi = torch.max(min_x1, min_x2)
    max_yi = torch.min(max_y1, max_y2)
    min_yi = torch.max(min_y1, min_y2)
    inter = (max_xi - min_xi).clamp(0) * (max_yi - min_yi).clamp(0)
    union = bbox1[..., 2:3] * bbox1[..., 3:4] + bbox2[..., 2:3] * bbox2[..., 3:4] - inter
    return inter / (union + 1e-6)

# Compute the bounding box (max_x, min_x, max_y, min_y) for an ellipse that has 
# center (h,k), semimajor axis a and semiminor axis b, and is rotated through angle phi.
def compute_bbox_ellipse(a, b, phi, h, k):
    t = np.arctan(-b*np.tan(phi)/(a+1e-6))
    max_x = h + a*np.cos(t)*np.cos(phi) - b*np.sin(t)*np.sin(phi)
    min_x = h + a*np.cos(t + np.pi)*np.cos(phi) - b*np.sin(t + np.pi)*np.sin(phi)
    if max_x < min_x:
        max_x, min_x = min_x, max_x
    t = np.arctan(b*(1/np.tan(phi))/(a+1e-6))
    max_y = k + b*np.sin(t)*np.cos(phi) + a*np.cos(t)*np.sin(phi)
    min_y = k + b*np.sin(t + np.pi)*np.cos(phi) + a*np.cos(t + np.pi)*np.sin(phi)
    if max_y < min_y:
        max_y, min_y = min_y, max_y
    return max_x, min_x, max_y, min_y

def boxes_to_yoloLabel(imheight, imwidth, boxes, S, B, C):
    label = torch.zeros((S, S, C + B * 5))
    for box in boxes:
        x, y, w, h = box
        # determine which cell the box is in
        i, j = int(S * y / imheight), int(S * x / imwidth)
        # calculate relative position and size of the box with respect to the cell
        x_cell, y_cell = S * x / imwidth - j, S * y / imheight - i
        w_cell, h_cell = S * w / imwidth, S * h / imheight
        
        if label[i, j, C] == 0:
            label[i, j, C] = 1
            label[i, j, C+1:C+5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
    return label

def yoloLabel_to_boxes(imheight, imwidth, label, S, B, C):
    boxes = []
    for i in range(S):
        for j in range(S):
            if label[i, j, C] >= 0.5:
                p_c, x_cell, y_cell, w_cell, h_cell = label[i, j, C:C+5].tolist()
                x, y = (x_cell + j) * imwidth / S, (y_cell + i) * imheight / S
                w, h = w_cell * imwidth / S, h_cell * imheight / S
                boxes.append([p_c, x, y, w, h])
            if label[i, j, C+5] >= 0.5:
                p_c, x_cell, y_cell, w_cell, h_cell = label[i, j, C+5:C+10].tolist()
                x, y = (x_cell + j) * imwidth / S, (y_cell + i) * imheight / S
                w, h = w_cell * imwidth / S, h_cell * imheight / S
                boxes.append([p_c, x, y, w, h])
    return boxes

def nonMaximumSuppression(bboxes, thres_iou):
    final_bboxes = []
    bboxes = np.array(bboxes)
    bboxes = bboxes[bboxes[:,0].argsort()]
    while len(bboxes) > 0:
        # select the bbox with highest confidence score and put it into final_bboxes
        selected_bbox = bboxes[-1]
        final_bboxes.append(selected_bbox)
        bboxes = bboxes[:-1]
        # compare it with all other unselected bboxes by calculating their IoU
        _, x, y, w, h = selected_bbox
        min_x1 = x - 0.5 * w
        min_y1 = y - 0.5 * h
        max_x1 = min_x1 + w
        max_y1 = min_y1 + h
        updated_bboxes = []
        for bbox in bboxes:
            _, x, y, w, h = bbox
            min_x2 = x - 0.5 * w
            min_y2 = y - 0.5 * h
            max_x2 = min_x2 + w
            max_y2 = min_y2 + h
            # find the intersection box
            max_xi, min_xi, max_yi, min_yi = min(max_x1, max_x2), max(min_x1, min_x2), min(max_y1, max_y2), max(min_y1, min_y2)
            wi = max_xi - min_xi
            hi = max_yi - min_yi
            if wi <= 0 or hi <= 0:
                updated_bboxes.append(bbox)
                continue
            # calculate intersection area
            inter = wi * hi
            union = (max_x1 - min_x1) * (max_y1 - min_y1) + (max_x2 - min_x2) * (max_y2 - min_y2) - inter
            IoU = inter / union
            if IoU < thres_iou:
                updated_bboxes.append(bbox)
                # discard the bbox that overlaps too much with the selected bbox
        bboxes = updated_bboxes
    return final_bboxes

def evaluateFaceDetection(y_pred, y_ground_truth, thres_iou):
    tp, fp, fn = 0, 0, 0
    for i, bboxes in enumerate(y_pred):
        num_correct = 0
        for bbox in bboxes:
            # calculate if this bbox is a true positive
            max_x1, min_x1, max_y1, min_y1 = bbox
            correct = False
            for true_bbox in y_ground_truth[i]:
                max_x2, min_x2, max_y2, min_y2 = true_bbox
                # find the intersection box
                max_xi, min_xi, max_yi, min_yi = min(max_x1, max_x2), max(min_x1, min_x2), min(max_y1, max_y2), max(min_y1, min_y2)
                w = max_xi - min_xi
                h = max_yi - min_yi
                if w <= 0 or h <= 0:
                    continue
                # calculate intersection area
                inter = w * h
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
        fn += len(y_ground_truth[i]) - num_correct
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

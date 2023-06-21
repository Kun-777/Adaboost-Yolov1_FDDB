import numpy as np
import cv2
from tqdm import tqdm

def faceDetection(X, face_classifier, scales, delta, score_thres, thres_iou):
    all_faces = []
    for img in tqdm(X):
        sub_windows = []
        sub_window_bbox = []
        for scale in scales:
            # step of the scanner is calculated by round(scale*delta)
            # scan from left to right, top to bottom
            for y in [int(round(scale*delta)*i) for i in range(img.shape[0] // round(scale*delta))]:
                for x in [int(round(scale*delta)*i) for i in range(img.shape[1] // round(scale*delta))]:
                    sub_window = cv2.resize(img[y:y+scale, x:x+scale], (24,24))
                    sub_windows.append(sub_window)
                    sub_window_bbox.append([x+scale, x, y+scale, y])
            # scan from right to left, top to bottom
            for y in [int(round(scale*delta)*i) for i in range(img.shape[0] // round(scale*delta))]:
                for x in [int(img.shape[1] - round(scale*delta)*i) for i in range(1, (img.shape[1] // round(scale*delta)) + 1)]:
                    sub_window = cv2.resize(img[y:y+scale, x:x+scale], (24,24))
                    sub_windows.append(sub_window)
                    sub_window_bbox.append([x+scale, x, y+scale, y])
            # scan from left to right, bottom to top
            for y in [int(img.shape[0] - round(scale*delta)*i) for i in range(1, (img.shape[0] // round(scale*delta)) + 1)]:
                for x in [int(round(scale*delta)*i) for i in range(img.shape[1] // round(scale*delta))]:
                    sub_window = cv2.resize(img[y:y+scale, x:x+scale], (24,24))
                    sub_windows.append(sub_window)
                    sub_window_bbox.append([x+scale, x, y+scale, y])
            # scan from right to left, bottom to top
            for y in [int(img.shape[0] - round(scale*delta)*i) for i in range(1, (img.shape[0] // round(scale*delta)) + 1)]:
                for x in [int(img.shape[1] - round(scale*delta)*i) for i in range(1, (img.shape[1] // round(scale*delta)) + 1)]:
                    sub_window = cv2.resize(img[y:y+scale, x:x+scale], (24,24))
                    sub_windows.append(sub_window)
                    sub_window_bbox.append([x+scale, x, y+scale, y])
        # normalize all sub-windows
        img = np.array(sub_windows)
        mean = img.mean(1).mean(1)
        squared_sum = np.power(img,2).sum(1).sum(1)
        variance = squared_sum/(img.shape[1]*img.shape[2]) - mean**2
        mean = np.repeat(mean, img.shape[1]*img.shape[2]).reshape(img.shape)
        epsilon = 1e-9
        variance = np.repeat(variance + epsilon, img.shape[1]*img.shape[2]).reshape(img.shape)
        img = (img - mean) / variance
        # classify all subwindows
        pred, score = face_classifier.predict(img)
        # nms
        faces = nonMaximumSuppression(np.array(sub_window_bbox)[(pred == 1) & (score > score_thres)], score[(pred == 1) & (score > score_thres)], thres_iou)
        all_faces.append(faces)
    return all_faces


def nonMaximumSuppression(bboxes, scores, thres_iou):
    final_bboxes = []
    order = np.argsort(scores)
    scores = scores[order]
    bboxes = bboxes[order]
    while len(bboxes) > 0:
        # select the bbox with highest confidence score and put it into final_bboxes
        selected_bbox = bboxes[-1]
        final_bboxes.append(selected_bbox)
        bboxes = bboxes[:-1]
        # compare it with all other unselected bboxes by calculating their IoU
        max_x1, min_x1, max_y1, min_y1 = selected_bbox
        updated_bboxes = []
        for bbox in bboxes:
            max_x2, min_x2, max_y2, min_y2 = bbox
            # find the intersection box
            max_xi, min_xi, max_yi, min_yi = min(max_x1, max_x2), max(min_x1, min_x2), min(max_y1, max_y2), max(min_y1, min_y2)
            w = max_xi - min_xi
            h = max_yi - min_yi
            if w <= 0 or h <= 0:
                updated_bboxes.append(bbox)
                continue
            # calculate intersection area
            inter = w * h
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


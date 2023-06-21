import glob
import numpy as np
from PIL import Image
import random
import cv2
from tqdm import tqdm

def dataset_for_classification(raw_data, num_neg_samples):
    dataset = {'img':[], 'labels':[]}
    for i, raw_img in enumerate(raw_data['img']):
        # Crop face bboxs into squares as positive samples
        for max_x, min_x, max_y, min_y in raw_data['faces'][i]:
            h, w = max_y - min_y, max_x - min_x
            if h > w:
                # cut top and bottom
                min_y += (h - w) // 2
                max_y -= (h - w) // 2
            else:
                # cut left and right
                min_x += (w - h) // 2
                max_x -= (w - h) // 2
            img_patch = raw_img[min_y:max_y, min_x:max_x]
            img_patch = cv2.resize(img_patch, (24,24))
            # add its vertical mirror image (what viola-jones did)
            dataset['img'].append(img_patch)
            dataset['img'].append(img_patch[:,::-1])
            dataset['labels'].append(1)
            dataset['labels'].append(1)

        # now generate non-face patches as negative samples
        # we consider a patch with faces that are too big or too small as non faces
        count = 0
        while (count < num_neg_samples):
            patch_size = random.randint(24, min(raw_img.shape[0]//2, raw_img.shape[1]//2))
            # Let x, y be the top left corner of the image patch
            y, x = random.randint(0,raw_img.shape[0]-patch_size), random.randint(0,raw_img.shape[1]-patch_size)
            isface = False
            for max_x, min_x, max_y, min_y in raw_data['faces'][i]:
                # check if patch contains a face
                if y < min_y + (max_y - min_y) / 4 and y + patch_size > max_y - (max_y - min_y) / 4 \
                        and x < min_x + (max_x - min_x) / 4 and x + patch_size > max_x - (max_x - min_x) / 4 \
                        and y > min_y - (max_y - min_y) / 4 and y + patch_size < max_y + (max_y - min_y) / 4 \
                        and x > min_x - (max_x - min_x) / 4 and x + patch_size < max_x + (max_x - min_x) / 4:
                    isface = True
                    break
            if isface:
                continue
            img_patch = raw_img[y:y+patch_size, x:x+patch_size]
            img_patch = cv2.resize(img_patch, (24,24))
            dataset['img'].append(img_patch)
            dataset['labels'].append(0)
            count += 1
    # normalize each patch by variance to minimize the effect of different lighting conditions
    img = np.array(dataset['img'])
    mean = img.mean(1).mean(1)
    squared_sum = np.power(img,2).sum(1).sum(1)
    variance = squared_sum/(img.shape[1]*img.shape[2]) - mean**2
    mean = np.repeat(mean, img.shape[1]*img.shape[2]).reshape(img.shape)
    epsilon = 1e-9
    variance = np.repeat(variance + epsilon, img.shape[1]*img.shape[2]).reshape(img.shape)
    img = (img - mean) / variance
    dataset['img'] = list(img)
    return dataset


def dataloader(FDDB_folds_dir, originalPics_dir):
    train_data = {'img':[], 'faces':[]}
    test_data = {'img':[], 'faces':[]}
    for fold in glob.glob(FDDB_folds_dir + "*[1-8]-ellipseList.txt"):
        with open(fold) as f:
            img_path = f.readline()[:-1]
            while img_path != "":
                img = np.asarray(Image.open(originalPics_dir + img_path + ".jpg"))
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                train_data['img'].append(img)
                num_faces = int(f.readline()[:-1])
                faces = np.zeros([num_faces, 4])
                for i in range(num_faces):
                    major_axis_radius, minor_axis_radius, angle, center_x, center_y, valid = f.readline()[:-1].split()
                    assert(valid == '1')
                    max_x, min_x, max_y, min_y = compute_bbox_ellipse(float(major_axis_radius), float(minor_axis_radius), float(angle), float(center_x), float(center_y))
                    max_x, min_x, max_y, min_y = min(img.shape[1]-1, max_x), max(0, min_x), min(img.shape[0]-1, max_y), max(0, min_y)
                    faces[i, :] = max_x, min_x, max_y, min_y
                train_data['faces'].append(faces.astype(int))
                img_path = f.readline()[:-1]
    for fold in glob.glob(FDDB_folds_dir + "*09-ellipseList.txt") + glob.glob(FDDB_folds_dir + "*10-ellipseList.txt"):
        with open(fold) as f:
            img_path = f.readline()[:-1]
            while img_path != "":
                img = np.asarray(Image.open(originalPics_dir + img_path + ".jpg"))
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                test_data['img'].append(img)
                num_faces = int(f.readline()[:-1])
                faces = np.zeros([num_faces, 4])
                for i in range(num_faces):
                    major_axis_radius, minor_axis_radius, angle, center_x, center_y, valid = f.readline()[:-1].split()
                    assert(valid == '1')
                    max_x, min_x, max_y, min_y = compute_bbox_ellipse(float(major_axis_radius), float(minor_axis_radius), float(angle), float(center_x), float(center_y))
                    max_x, min_x, max_y, min_y = min(img.shape[1]-1, max_x), max(0, min_x), min(img.shape[0]-1, max_y), max(0, min_y)
                    faces[i, :] = max_x, min_x, max_y, min_y
                test_data['faces'].append(faces.astype(int))
                img_path = f.readline()[:-1]
    return train_data, test_data

# Compute the bounding box (max_x, min_x, max_y, min_y) for an ellipse that has 
# center (h,k), semimajor axis a and semiminor axis b, and is rotated through angle phi.
def compute_bbox_ellipse(a, b, phi, h, k):
    t = np.arctan(-b*np.tan(phi)/a)
    max_x = h + a*np.cos(t)*np.cos(phi) - b*np.sin(t)*np.sin(phi)
    min_x = h + a*np.cos(t + np.pi)*np.cos(phi) - b*np.sin(t + np.pi)*np.sin(phi)
    if max_x < min_x:
        max_x, min_x = min_x, max_x
    t = np.arctan(b*(1/np.tan(phi))/a)
    max_y = k + b*np.sin(t)*np.cos(phi) + a*np.cos(t)*np.sin(phi)
    min_y = k + b*np.sin(t + np.pi)*np.cos(phi) + a*np.cos(t + np.pi)*np.sin(phi)
    if max_y < min_y:
        max_y, min_y = min_y, max_y
    return max_x, min_x, max_y, min_y

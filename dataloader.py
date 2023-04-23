import glob
import numpy as np
from PIL import Image
import random
import cv2
import tqdm

def dataset_for_classification(train_raw, test_raw, num_samples_per_img):
    train_data = {'img':[], 'labels':[]}
    test_data = {'img':[], 'labels':[]}
    for i, raw_img in enumerate(train_raw['img']):
        if len(raw_img.shape) == 3:
            raw_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)

        # TODO: Redo this
        possible_coordinates = [(y, x) for y in range(raw_img.shape[0]-48) for x in range(raw_img.shape[1]-48)]
        sample = random.sample(possible_coordinates, num_samples_per_img)
        for y, x in sample:
            # Let x, y be the top left corner of the image patch
            patch_size = random.randint(48, min(raw_img.shape[0]-y, raw_img.shape[1]-x))
            img_patch = raw_img[y:y+patch_size, x:x+patch_size]
            img_patch = cv2.resize(img_patch, (24,24))
            # check if patch contains face, we consider it contains a face if it covers 75% of the bounding box of a face
            label = 0
            for max_x, min_x, max_y, min_y in train_raw['faces'][i]:
                if y < min_y + (max_y - min_y) / 8 and y + patch_size > max_y - (max_y - min_y) / 8 \
                        and x < min_x + (max_x - min_x) / 8 and x + patch_size > max_x - (max_x - min_x) / 8:
                    label = 1
                    break
            train_data['img'].append(img_patch)
            train_data['labels'].append(label)
    for i, raw_img in enumerate(test_raw['img']):
        if len(raw_img.shape) == 3:
            raw_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)
        possible_coordinates = [(y, x) for y in range(raw_img.shape[0]-48) for x in range(raw_img.shape[1]-48)]
        sample = random.sample(possible_coordinates, num_samples_per_img)
        for y, x in sample:
            # Let x, y be the top left corner of the image patch
            patch_size = random.randint(48, min(raw_img.shape[0]-y, raw_img.shape[1]-x))
            img_patch = raw_img[y:y+patch_size, x:x+patch_size]
            img_patch = cv2.resize(img_patch, (24,24))
            # check if patch contains face, we consider it contains a face if it covers 75% of the bounding box of a face
            label = 0
            for max_x, min_x, max_y, min_y in train_raw['faces'][i]:
                if y < min_y + (max_y - min_y) / 8 and y + patch_size > max_y - (max_y - min_y) / 8 \
                        and x < min_x + (max_x - min_x) / 8 and x + patch_size > max_x - (max_x - min_x) / 8:
                    label = 1
                    break
            test_data['img'].append(img_patch)
            test_data['labels'].append(label)
    return train_data, test_data




def dataloader(FDDB_folds_dir, originalPics_dir):
    train_data = {'img':[], 'faces':[]}
    test_data = {'img':[], 'faces':[]}
    for fold in glob.glob(FDDB_folds_dir + "*[1-8]-ellipseList.txt"):
        with open(fold) as f:
            img_path = f.readline()[:-1]
            while img_path != "":
                img = np.asarray(Image.open(originalPics_dir + img_path + ".jpg"))
                train_data['img'].append(img)
                num_faces = int(f.readline()[:-1])
                faces = np.zeros([num_faces, 4])
                for i in range(num_faces):
                    major_axis_radius, minor_axis_radius, angle, center_x, center_y, valid = f.readline()[:-1].split()
                    assert(valid == '1')
                    faces[i, :] = compute_bbox_ellipse(float(major_axis_radius), float(minor_axis_radius), float(angle), float(center_x), float(center_y))
                train_data['faces'].append(faces)
                img_path = f.readline()[:-1]
    for fold in glob.glob(FDDB_folds_dir + "*09-ellipseList.txt") + glob.glob(FDDB_folds_dir + "*10-ellipseList.txt"):
        with open(fold) as f:
            img_path = f.readline()[:-1]
            while img_path != "":
                img = np.asarray(Image.open(originalPics_dir + img_path + ".jpg"))
                test_data['img'].append(img)
                num_faces = int(f.readline()[:-1])
                faces = np.zeros([num_faces, 4])
                for i in range(num_faces):
                    major_axis_radius, minor_axis_radius, angle, center_x, center_y, valid = f.readline()[:-1].split()
                    assert(valid == '1')
                    faces[i, :] = compute_bbox_ellipse(float(major_axis_radius), float(minor_axis_radius), float(angle), float(center_x), float(center_y))
                test_data['faces'].append(faces)
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
    return np.array([max_x, min_x, max_y, min_y])

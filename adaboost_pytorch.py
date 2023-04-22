import numpy as np
import cv2
import random
from tqdm import tqdm
import torch

class MulticlassClassifier:
    def __init__(self, classes, imsize):
        self.classes = classes
        # randomly sample 20 locations to extract haar features
        possible_coordinates = [(x, y) for x in range(1,imsize[0]+1) for y in range(1,imsize[1]+1)]
        self.sample = random.sample(possible_coordinates, 20)

    def fit(self, X, labels, T):
        # initialize a list of one-vs-all classifiers
        self.ova_clfs = []
        X = haarFeatures(X, self.sample)
        for i in self.classes:
            y = (labels==i).astype(int)
            m = np.count_nonzero(y==0)
            l = len(y) - m
            # initialize weights
            w = (1-y) / (2*m) + y / (2*l)
            # print("Adaboost for class", i)
            # one-vs-all classifier
            ova_clf = AdaboostClassifier(T)
            ova_clf.fit(X, y, w)
            self.ova_clfs.append(ova_clf)
    
    def predict(self, X):
        # extract Haar features the same way as we train
        X = haarFeatures(X, self.sample)
        # compare prediction scores from all one-vs-all predictions
        scores = []
        for ova_clf in self.ova_clfs:
            _, score = ova_clf.predict(X)
            scores.append(score)
        y_pred = np.argmax(np.array(scores), axis=0)
        return y_pred

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        multiclass_acc = (y == y_pred).sum(0)/len(y)
        singleclass_accs = []
        X = haarFeatures(X, self.sample)
        for i, ova_clf in enumerate(self.ova_clfs):
            y_class = (y==i).astype(int)
            singleclass_accs.append(ova_clf.evaluate(X, y_class))
        return multiclass_acc, singleclass_accs

# num_wk_clfs should be a list of number of weak classifiers to try
def cross_validate_adaboost(X_train, y_train, X_test, y_test, num_folds, num_wk_clfs):
    # Split the data into num_folds folds
    folds_X = np.array_split(X_train, num_folds)
    folds_y = np.array_split(y_train, num_folds)
    adaboost_classifier = MulticlassClassifier(10, [32,32])
    avg_accs = []
    for T in num_wk_clfs:
        avg_acc = 0
        # Perform cross-validation for each T
        for i in tqdm(range(num_folds)):
            X_train_fold = np.concatenate([fold for j, fold in enumerate(folds_X) if j != i])
            y_train_fold = np.concatenate([fold for j, fold in enumerate(folds_y) if j != i])
            X_val = folds_X[i]
            y_val = folds_y[i]
            adaboost_classifier.fit(X_train_fold, y_train_fold, T)
            accuracy, _ = adaboost_classifier.evaluate(X_val, y_val)
            avg_acc += accuracy/num_folds
        avg_accs.append(avg_acc)
    best_T = num_wk_clfs[np.array(avg_accs).argmax()]
    adaboost_classifier.fit(X_train, y_train, T)
    final_acc = adaboost_classifier.evaluate(X_test, y_test)
    return best_T, final_acc



def adaboost_test(train_set, test_set, T):
    # randomly sample 25 locations to extract haar features
    possible_coordinates = [(x, y) for x in range(1,33) for y in range(1,33)]
    sample = random.sample(possible_coordinates, 20)
    X_train = haarFeatures(np.array(train_set['data']), sample)
    X_test = haarFeatures(np.array(test_set['data']), sample)
    train_labels = np.array(train_set['labels'])
    test_labels = np.array(test_set['labels'])
    adaboost_clfs = []
    train_errors = None
    test_errors = None
    for i in range(len(np.unique(train_labels))):
        y = (train_labels==i).astype(int)
        y_test = (test_labels==i).astype(int)
        m = np.count_nonzero(y==0)
        l = len(y) - m
        # initialize weights
        w = (1-y) / (2*m) + y / (2*l)
        print("Adaboost for class", i)
        # one-vs-all classifier
        ova_clf = AdaboostClassifier(T)
        if i == 0:
            # keep track of train and test errors for validation
            train_errors, test_errors = ova_clf.fit(X_train, y, w, X_test, y_test)
        else:
            ova_clf.fit(X_train, y, w)

        # get accuracy for each one-vs-all classifier
        y_pred_test, _ = ova_clf.predict(X_test)
        acc = (1 - np.abs(y_pred_test - y_test)).sum(0)/len(y_test)
        adaboost_clfs.append((ova_clf, acc))
        break
    
    return adaboost_clfs, train_errors, test_errors

# extract haar features from a randomly sampled set of points on a set of images
# im is input images of shape (num_images, size of image flattened)
# sample is a list of points (x,y) indicates the coordinates of pixel to extract feature from
# returns an array of features of shape (num_images, num_features)
def haarFeatures(im, sample):
    # precompute integral image
    grayim = []
    if len(im.shape) == 4:
        # rgb img
        for i in range(im.shape[0]):
            grayim.append(cv2.cvtColor(im[i],cv2.COLOR_BGR2GRAY))
    else:
        grayim = im
    iim = integralIm(np.array(grayim))
    # get exhausive set of haar features
    haar_features = []
    H = iim.shape[1]
    W = iim.shape[2]
    for h in range(2,6):
        for w in range(2,6):
            # rectangular haar filter of size h x w
            for i, j in sample:
                # 2 rectangles that are horizontally adjacent
                if j+w < W:
                    haar_features.append((iim[:,i,j] + iim[:,i-h,j-w] - iim[:,i-h,j] - iim[:,i,j-w])
                            - (iim[:,i,j+w] + iim[:,i-h,j] - iim[:,i-h,j+w] - iim[:,i,j])
                            )
                # 2 rectangles that are vertcally adjacent
                if i+h < H:
                    haar_features.append((iim[:,i,j] + iim[:,i-h,j-w] - iim[:,i-h,j] - iim[:,i,j-w])
                            - (iim[:,i+h,j] + iim[:,i,j-w] - iim[:,i,j] - iim[:,i+h,j-w])
                            )
                # 3 rectangles that are horizontally adjacent
                if j+2*w < W:
                    haar_features.append((iim[:,i,j] + iim[:,i-h,j-w] - iim[:,i-h,j] - iim[:,i,j-w])
                            - (iim[:,i,j+w] + iim[:,i-h,j] - iim[:,i-h,j+w] - iim[:,i,j]) 
                            + (iim[:,i,j+2*w] + iim[:,i-h,j+w] - iim[:,i-h,j+2*w] - iim[:,i,j+w])
                            )
                # 3 rectangles that are vertically adjacent
                if i+2*h < H:
                    haar_features.append((iim[:,i,j] + iim[:,i-h,j-w] - iim[:,i-h,j] - iim[:,i,j-w])
                            - (iim[:,i+h,j] + iim[:,i,j-w] - iim[:,i,j] - iim[:,i+h,j-w])
                            + (iim[:,i+2*h,j] + iim[:,i+h,j-w] - iim[:,i+h,j] - iim[:,i+2*h,j-w])
                            )
                # 4 rectangles
                if j+w < W and i+h < H:
                    haar_features.append((iim[:,i,j] + iim[:,i-h,j-w] - iim[:,i-h,j] - iim[:,i,j-w])
                            - (iim[:,i,j+w] + iim[:,i-h,j] - iim[:,i-h,j+w] - iim[:,i,j])
                            - (iim[:,i+h,j] + iim[:,i,j-w] - iim[:,i,j] - iim[:,i+h,j-w])
                            + (iim[:,i+h,j+w] + iim[:,i,j] - iim[:,i,j+w] - iim[:,i+h,j])
                            )
    return np.array(haar_features).swapaxes(0,1)

# im is n x 32 x 32 x 3
def integralIm(im):
    iim = np.zeros([im.shape[0], im.shape[1]+1, im.shape[2]+1])
    for x in range(im.shape[1]):
        for y in range(im.shape[2]):
            iim[:,x+1,y+1] = iim[:,x,y+1] + iim[:,x+1,y] - iim[:,x,y] + im[:,x,y]
    # iim = np.delete(np.delete(iim, 0, 1), 0, 2)
    return iim

class WeakClassifier:
    def __init__(self):
        pass

    # select the best feature, threshold and sign to split the data
    # X is a data matrix with shape (num_samples, num_features)
    # y is a vector of labels with shape (num_samples,)
    # w is a weight vector with shape (num_samples,)
    # returns the error of the classifier that uses the best feature and best threshold
    # as well as e, a vector indicates if each sample is classified correctly
    def fit(self, X, y, w):
        with torch.no_grad():
            self.best_feature = None
            self.best_threshold = None
            W = torch.tile(w, (X.shape[1],1)).transpose(0,1)
            Y = torch.tile(y, (X.shape[1],1)).transpose(0,1)
            # sort each feature column and reorder the Weights and Labels in the same order
            sorted_idx = torch.argsort(X, dim=0)
            sorted_X = torch.take_along_dim(X, sorted_idx, dim=0)
            sorted_W = torch.take_along_dim(W, sorted_idx, dim=0)
            sorted_Y = torch.take_along_dim(Y, sorted_idx, dim=0)
            # compute cumulative weighted error and find minimum instead of trying every threshold
            # each column of cum_w_error_left is [weighted error for predicting all instances 
            # with indices smaller than or equal to i to be 1 for all i]
            cum_w_error_left = torch.cumsum(sorted_W * (sorted_Y ^ 1), dim=0)[:-1, :]
            cum_w_error_right = torch.cumsum(torch.flip(sorted_W, [0]) * torch.flip(sorted_Y, [0]), dim=0).flip([0])[1:, :]
            cum_w_error = cum_w_error_left + cum_w_error_right
            parity = torch.ones(cum_w_error.shape)
            # flip parity if error > 0.5
            need_to_flip = cum_w_error > 0.5
            cum_w_error[need_to_flip] = 1 - cum_w_error[need_to_flip]
            parity[need_to_flip] = -1
            # find the index of the minimum error
            thres_idx, self.best_feature = np.unravel_index(int(cum_w_error.argmin()), cum_w_error.shape)
            self.parity = int(parity[thres_idx, self.best_feature])
            self.best_threshold = float((sorted_X[thres_idx, self.best_feature] + sorted_X[thres_idx+1, self.best_feature]) / 2)
            # calculate e based on the best feature and threshold we found
            e = (X[:, self.best_feature] < self.best_threshold) ^ y
            best_error = (w*e).sum(0)
            if self.parity == -1:
                best_error = 1 - best_error
                e = 1 - e
        return float(best_error), e
            
    def predict(self, X):
        if self.parity == 1:
            return (X[:, self.best_feature] < self.best_threshold).astype(int)
        else:
            return (X[:, self.best_feature] >= self.best_threshold).astype(int)
        
class AdaboostClassifier:
    def __init__(self, T):
        self.T = T
        self.weak_clfs = []
    
    # returns train and test errors at each step if test set is given
    def fit(self, X_train, y_train, w, X_test=None, y_test=None):
        train_errors = []
        test_errors = []
        X_train, y_train, w = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(w)
        if torch.cuda.is_available():
            X_train, y_train, w = X_train.to('cuda'), y_train.to('cuda'), w.to('cuda')
        for _ in tqdm(range(self.T)):
            # normalize weight
            w = w/(w.sum(0))
            # Choose the best feature to split the data
            weak_clf = WeakClassifier()
            error, e = weak_clf.fit(X_train, y_train, w)
            w = w * ((error/(1-error)) ** (1-e))
            alpha = np.log(1/(error/(1-error)))
            self.weak_clfs.append((weak_clf, alpha))
            if X_test is not None:
                # calculate train and test errors
                train_pred, _ = self.predict(X_train.cpu().numpy())
                test_pred, _ = self.predict(X_test)
                train_errors.append((np.abs(train_pred - y_train.cpu().numpy())).sum(0)/len(train_pred))
                test_errors.append((np.abs(test_pred - y_test)).sum(0)/len(test_pred))
        return train_errors, test_errors
    
    def predict(self, X):
        sum_ah = np.zeros(X.shape[0])
        sum_alpha = 0
        # self.weak_clfs is a list of tuples (WeakClassifier, alpha)
        for weak_clf, alpha in self.weak_clfs:
            y_pred = weak_clf.predict(X)
            sum_ah += y_pred * alpha
            sum_alpha += alpha
        score = sum_ah/sum_alpha
        return (sum_ah >= 0.5 * sum_alpha).astype(int), score
    
    def evaluate(self, X, y):
        y_pred, _ = self.predict(X)
        return (y == y_pred).sum(0)/len(y)


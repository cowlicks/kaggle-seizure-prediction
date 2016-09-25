'''
In this file we demonstrate using dask to load the data and compute our
features.  Then we train an SVM and test its predictions against the data.
'''
from glob import glob

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
from scipy.io import loadmat
from sklearn import cross_validation
from sklearn.svm import SVC


count = 0

def arr_from_mat(fn):
    return loadmat(fn, struct_as_record=False)['dataStruct'][0, 0].data.T

def da_from_one_mat(path):
    global count
    name = 'mat-da' + str(count)
    count += 1
    chunks = ((16,), (240000,))
    return da.Array({(name, 0, 0): (arr_from_mat, path)}, name, chunks)

def da_from_mats(pattern):
    ''' Convert datafiles matching pattern to a dask array.'''
    paths = sorted(glob(pattern))
    return da.stack([da_from_one_mat(p) for p in paths])

def feature(data):
    return da.mean(data**2, axis=2)**0.5
    
def features_from_mats(pattern):
    '''
    Get features from files matching a given pattern.
    '''
    da = da_from_mats(pattern)
    return feature(da)


if __name__ == '__main__':
    # load the data
    baseline = features_from_mats('../data/train_1/1_*_0.mat')
    preictal = features_from_mats('../data/train_1/1_*_1.mat')

    with ProgressBar():
        X = da.concatenate([baseline, preictal]).compute()

    y = np.concatenate([np.zeros(baseline.shape[0]),  # baseline = 0
                        np.ones(preictal.shape[0])])  # preictal = 1

    clf = SVC(class_weight='balanced')

    scores = cross_validation.cross_val_score(clf, X, y, n_jobs=-1)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # Accuracy: 0.89 (+/- 0.00)

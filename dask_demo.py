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

def mat_to_arr(fn):    
    return loadmat(fn, struct_as_record=False)['dataStruct'][0, 0].data.T

def one_mat_to_da(path):
    global count
    name = 'mat-da' + str(count)
    count += 1
    chunks = ((16,), (240000,))
    return da.Array({(name, 0, 0): (mat_to_arr, path)}, name, chunks)

def mat_to_da(pattern):
    ''' Convert datafiles matching pattern to a dask array.'''
    paths = sorted(glob(pattern))
    return da.stack([one_mat_to_da(p) for p in paths])

def feature(data):
    return data.var(axis=2)
    
def features_from_mats(pattern):
    '''
    Get features from files matching a given pattern.
    '''
    da = mat_to_da(pattern)
    return feature(da)


if __name__ == '__main__':
    # load the data
    baseline = features_from_mats('../data/train_1/1_*_0.mat')
    preictal = features_from_mats('../data/train_1/1_*_1.mat')

    with ProgressBar():
        X = da.concatenate([baseline, preictal]).compute()

    y = np.concatenate([np.zeros(baseline.shape[0]),  # baseline = 0
                        np.ones(preictal.shape[0])])  # preictal = 1

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=0.4, random_state=0)

    clf = SVC()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))  # 0.8733!!!

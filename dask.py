'''
In this file we demonstrate using dask to load the data and compute our
features.  Then we train an SVM and test its predictions against the data.
'''

import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
from glob import glob
from scipy.io import loadmat
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

def partition(data, step):
    '''
    partition the data into a training and test set. Every stepth row goes into
    the test set.
    '''
    nsamps = data.shape[0]
    frac = nsamps // step
    remainder = nsamps % step
    test = data[::step]
    train = da.concatenate([data[step*i:step*(i + 1) - 1] for i in range(frac)])
    if remainder != 0:
        train = da.concatenate([train, data[-remainder:]])
    return train, test
    
def features_from_mats(pattern, step):
    '''
    Get features from files matching a given pattern.
    '''
    da = mat_to_da(pattern)
    featured = feature(da)
    train, test = partition(featured, step)
    return train, test


if __name__ == '__main__':
    # load the data
    preictal, testpre = features_from_mats('../data/train_1/1_*_1.mat', 4)
    baseline, testbase = features_from_mats('../data/train_1/1_*_0.mat', 4)

    # Make our training data
    X = da.concatenate([preictal, baseline])
    y = np.concatenate([np.zeros(preictal.shape[0]),
                        np.ones(baseline.shape[0])])

    clf = SVC()
    with ProgressBar():
        clf.fit(X.compute(), y)
        outpre = clf.predict(testpre.compute()).astype(int)
        outbase = clf.predict(testbase.compute()).astype(int)

    print("number wrong preictal %s " % (np.count_nonzero(outpre),))
    print("number wrong baseline %s " % (np.count_nonzero(outbase - 1),))

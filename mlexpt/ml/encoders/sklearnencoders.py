
import numpy as np
import joblib
from sklearn.decomposition import PCA, IncrementalPCA
from umap import UMAP
from tqdm import tqdm

from ..core import ExperimentalEncoder


class ExperimentalPCA(PCA, ExperimentalEncoder):
    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None):
        PCA.__init__(self, n_components=n_components, copy=copy, whiten=whiten,
                 svd_solver=svd_solver, tol=tol, iterated_power=iterated_power,
                 random_state=random_state)

    def fit(self, X, *args, **kwargs):
        PCA.fit(self, X, *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        return PCA.transform(self, X)

    def persist(self, path):
        joblib.dump(self, path)

    def trim(self):
        pass

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class ExperimentalIncrementalPCA(IncrementalPCA, ExperimentalEncoder):
    def __init__(self,
                 n_components=None,
                 whiten=False,
                 copy=True,
                 batch_size=None):
        IncrementalPCA.__init__(self,
                                n_components=n_components,
                                whiten=whiten,
                                copy=copy,
                                batch_size=batch_size)

    def fit(self, X, *args, **kwargs):
        IncrementalPCA.fit(self, X, *args, **kwargs)

    def partial_fit(self, X, y=None, check_input=True):
        IncrementalPCA.partial_fit(self, X, y=y, check_input=check_input)

    def fit_batch(self, dataset, *args, **kwargs):
        print('Fitting IncrementalPCA...')
        for batchid in tqdm(range(dataset.nbbatches)):
            X, _ = dataset.get_batch(batchid)
            X = np.array(X)
            IncrementalPCA.partial_fit(self, X, *args, **kwargs)

    def transform(self, X):
        IncrementalPCA.transform(self, X)

    def persist(self, path):
        joblib.dump(self, path)

    def trim(self):
        pass

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class ExperimentalUMAP(UMAP, ExperimentalEncoder):
    def fit(self, X, *args, **kwargs):
        UMAP.fit(self, X, *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        UMAP.transform(self, X)

    def persist(self, fileobj):
        pass

    def trim(self):
        pass

    @classmethod
    def load(cls, path):
        pass

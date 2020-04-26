
import numpy as np
import joblib
from sklearn.decomposition import PCA, IncrementalPCA
from umap import UMAP

from ..core import ExperimentalEncoder


class ExperimentalPCA(PCA, ExperimentalEncoder):
    def __init__(self, *args, **kwargs):
        PCA.__init__(self, *args, **kwargs)

    def fit(self, X, *args, **kwargs):
        PCA.fit(self, X, *args, **kwargs)

    def fit_batch(self, dataset, *args, **kwargs):
        x_tofit = None
        for fileid in range(dataset.nbfiles):
            X, _ = dataset.get_batch(fileid)
            x_tofit = np.array(X) if x_tofit is None else np.append(x_tofit, np.array(X))
        PCA.fit(x_tofit, *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        return PCA.transform(self, X)

    def transform_batch(self, dataset, *args, **kwargs):
        x_totransform = None
        for fileid in range(dataset.nbfiles):
            X, _ = dataset.get_batch(fileid)
            x_totransform = np.array(X) if x_totransform is None else np.append(x_totransform, np.array(X))
        return PCA.transform(x_totransform)

    def persist(self, path):
        joblib.dump(self, path)

    def trim(self):
        pass

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class ExperimentalIncrementalPCA(IncrementalPCA, ExperimentalEncoder):
    def __init__(self, *args, **kwargs):
        IncrementalPCA.__init__(self, *args, **kwargs)

    def fit(self, X, *args, **kwargs):
        IncrementalPCA.fit(self, X, *args, **kwargs)

    def partial_fit(self, X, y=None, check_input=True):
        IncrementalPCA.partial_fit(self, X, y=y, check_input=check_input)

    def fit_batch(self, dataset, *args, **kwargs):
        for fileid in range(dataset.nbfiles):
            X, _ = dataset.get_batch(fileid)
            X = np.array(X)
            IncrementalPCA.partial_fit(self, X, *args, **kwargs)

    def transform(self, X):
        IncrementalPCA.transform(self, X)

    def transform_batch(self, dataset, *args, **kwargs):
        x_totransform = None
        for fileid in range(dataset.nbfiles):
            X, _ = dataset.get_batch(fileid)
            X = np.array(X)
            x_totransform = np.array(X) if x_totransform is None else np.append(x_totransform, np.array(X))
        return IncrementalPCA.transform(self, x_totransform)

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


import joblib
from sklearn.decomposition import PCA
from umap import UMAP

from ..core import ExperimentalEncoder


class ExperimentalPCA(PCA, ExperimentalEncoder):
    def fit(self, X, *args, **kwargs):
        PCA.fit(self, X, *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        PCA.transform(self, X)

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

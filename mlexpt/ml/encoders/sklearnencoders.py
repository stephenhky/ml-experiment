
from sklearn.decomposition import PCA
from umap import UMAP

from ..core import ExperimentalEncoder


class ExperimentalPCA(PCA, ExperimentalEncoder):
    def __init__(self, *args, **kwargs):
        super(ExperimentalPCA, self).__init__(*args, **kwargs)

    def persist(self, fileobj):
        pass


class ExperimentalUMAP(UMAP, ExperimentalEncoder):
    def __init__(self, *args, **kwargs):
        super(ExperimentalUMAP, self).__init__(*args, **kwargs)

    def persist(self, fileobj):
        pass

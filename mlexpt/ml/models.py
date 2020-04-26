
from .classifiers.linear import MulticlassLogisticRegression, MulticlassBatchDatasetLogisticRegression

from .encoders.sklearnencoders import ExperimentalPCA, ExperimentalUMAP
from .encoders.dictembedding import DictEmbedding


classifiers_dict = {
    'LogisticRegression': MulticlassBatchDatasetLogisticRegression,
}


encoders_dict = {
    'PCA': ExperimentalPCA,
    'UMAP': ExperimentalUMAP,
    'embedding_dict': DictEmbedding
}


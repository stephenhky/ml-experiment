
from .classifiers.linear import MulticlassLogisticRegression

from .encoders.sklearnencoders import ExperimentalPCA, ExperimentalIncrementalPCA, ExperimentalUMAP
from .encoders.dictembedding import DictEmbedding


classifiers_dict = {
    'LogisticRegression': MulticlassLogisticRegression,
}


encoders_dict = {
    'PCA': ExperimentalPCA,
    'IncrementalPCA': ExperimentalIncrementalPCA,
    'UMAP': ExperimentalUMAP,
    'embedding_dict': DictEmbedding
}


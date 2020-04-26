
from .classifiers.linear import MulticlassLogisticRegression

from .encoders.sklearnencoders import ExperimentalPCA, ExperimentalUMAP
from .encoders.dictembedding import DictEmbedding


classifiers_dict = {
    'LogisticRegression': MulticlassLogisticRegression,
}


encoders_dict = {
    'PCA': ExperimentalPCA,
    'UMAP': ExperimentalUMAP,
    'embedding_dict': DictEmbedding
}


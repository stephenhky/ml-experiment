
import numpy as np

from ..core import ExperimentalEncoder


class DictEmbedding(ExperimentalEncoder):
    def __init__(self, feature2idx, embedding):
        self.feature2idx = feature2idx
        self.embedding = embedding
        self.target_dim = list(embedding.values())[0].shape[0]

        self.transform_matrix = np.zeros((len(feature2idx), self.target_dim))
        for key in embedding:
            if key in feature2idx:
                keyidx = feature2idx[key]
                self.transfom_matrix[keyidx, :] = embedding[key]

    def transform(self, X):
        return np.matmul(X, self.transform_matrix)



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

    def transform_batch(self, dataset, *args, **kwargs):
        transformed_x = None
        for fileid in range(dataset.nbfiles):
            X, _ = dataset.get_batch(fileid)
            X = np.array(X)
            this_transformed_x = np.matmul(X, self.transform_matrix)
            transformed_x = this_transformed_x if this_transformed_x is None else np.append(transformed_x, this_transformed_x, axis=0)
        return transformed_x

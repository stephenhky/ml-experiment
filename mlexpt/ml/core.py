
from abc import ABC

import numpy as np

class ExperimentalClassifier(ABC):
    def fit(self, X, Y, *args, **kwargs):
        pass

    def fit_batch(self, dataset, *args, **kwargs):
        pass

    def predict_proba(self, X, *args, **kwargs):
        pass

    def predict_proba_batch(self, dataset, *args, **kwargs):
        pass

    def persist(self, path):
        pass

    def trim(self):
        pass

    @classmethod
    def load(cls, path, *args, **kwargs):
        pass


class ExperimentalEncoder(ABC):
    def fit(self, X, *args, **kwargs):
        pass

    def fit_batch(self, dataset, *args, **kwargs):
        x_tofit = None
        for batchid in range(dataset.nbbatches):
            X, _ = dataset.get_batch(batchid)
            x_tofit = np.array(X) if x_tofit is None else np.append(x_tofit, np.array(X), axis=0)
        self.fit(x_tofit, *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        pass

    def transform_batch(self, dataset, *args, **kwargs):
        transformed_x = None
        for fileid in range(dataset.nbfiles):
            X, _ = dataset.get_batch(fileid)
            X = np.array(X)
            this_transformed_x = self.transform(X, *args, **kwargs)
            transformed_x = this_transformed_x if this_transformed_x is None else np.append(transformed_x, this_transformed_x, axis=0)
        return transformed_x

    def persist(self, path):
        pass

    def trim(self):
        pass

    @classmethod
    def load(cls, path, *args, **kwargs):
        pass


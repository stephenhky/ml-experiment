
from abc import ABC

class ExperimentalClassifier(ABC):
    def fit(self, X, Y, *args, **kwargs):
        pass

    def predict_proba(self, X, *args, **kwargs):
        pass

    def persist(self, path):
        pass

    def trim(self):
        pass


class ExperimentalEncoder(ABC):
    def fit(self, X, *args, **kwargs):
        pass

    def transform(self, X, *args, **kwargs):
        pass

    def persist(self, path):
        pass

    def trim(self):
        pass

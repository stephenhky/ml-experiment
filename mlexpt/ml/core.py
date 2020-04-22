
from abc import ABC

class ExperimentalClassifier(ABC):
    def fit(self, X, Y):
        pass

    def predict_proba(self, X):
        pass

    def persist(self, fileobj):
        pass

    def trim(self):
        pass


class ExperimentalEncoder(ABC):
    def fit(self, X, Y):
        pass

    def transform(self, X):
        pass

    def persist(self, fileobj):
        pass

    def trim(self):
        pass

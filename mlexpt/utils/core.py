
from collections import defaultdict

import numpy as np
from scipy.sparse import dok_matrix


def convert_listnum_ndarray(seq):
    if isinstance(seq[0], list):
        return np.array([convert_listnum_ndarray(item) for item in seq])
    else:
        return np.array(seq)


def iterate_categorical_values(datum, feature):
    if isinstance (datum.get(feature), list):
        for val in datum[feature]:
            if val is not None:
                yield val
    else:
        if datum.get(feature) is not None:
            yield datum[feature]


def generate_columndict(data_iterable, qual_features, binary_features, quant_features):
    #getting categorical values
    categorical_values_collection = defaultdict(lambda : set())
    for datum in data_iterable:
        for feature in qual_features:
            for val in iterate_categorical_values(datum, feature):
                categorical_values_collection[feature].add(val)
    categorical_values_collection = dict(categorical_values_collection)

    # generating dictionary
    feature2idx = {}
    idx2feature = {}
    counter = 0
    for feature in categorical_values_collection:
        print('\tQualitative Feature: {}, number of distinct of values: {}'.format(feature, len(categorical_values_collection[feature])))
        for val in categorical_values_collection[feature]:
            feature2idx[feature + ':' + str(val)] = counter
            counter += 1
    for feature in binary_features+ quant_features:
        print('\tBinary / Quantitative feature: {}'.format(feature))
        feature2idx[feature] = counter
        counter += 1

    idx2feature = {idx: feature for feature, idx in idx2feature.items()}

    return feature2idx, idx2feature


def convert_data_to_matrix(data, feature2idx, qual_features, binary_features, quant_features, labelcol, label2idx):
    X = dok_matrix((len(data), len(feature2idx)))
    Y = dok_matrix((len(data), len(label2idx)))

    for rowidx, datum in enumerate(data):
        for qual_feature in qual_features:
                datumval = datum[qual_feature]
                if not isinstance(datumval, list):
                    datumval = [datumval]
                for val in datumval:
                    colname = qual_feature + ':'+ str(val)
                    if colname in feature2idx:
                        colidx = feature2idx[colname]
                        X[rowidx, colidx] += 1
        for binary_feature in binary_features:
            colidx = feature2idx[binary_feature]
            X[rowidx, colidx] = float(datum[binary_feature])
        for quant_feature in quant_features:
            colidx = feature2idx[quant_feature]
            X[rowidx, colidx] = np.array(datum[quant_feature])

        if labelcol is not None and label2idx is not None:
            target_labels = datum[labelcol]
            if not isinstance(target_labels, list):
                target_labels = [target_labels]
            for label in target_labels:
                Y[rowidx, label2idx[labelcol+':'+label]] = 1

    return X, Y

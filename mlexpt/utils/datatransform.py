
from .core import generate_columndict, convert_data_to_matrix


def generate_columndict_withembeddings(data_iterable, qual_features, binary_features, quant_features, dimred_dict):
    reduced_qual_features = [feature for feature in qual_features if feature not in dimred_dict.keys()]
    feature2idx, idx2feature = generate_columndict(data_iterable, reduced_qual_features, binary_features, quant_features)
    for feature in dimred_dict:
        nb_cols_sofar = len(feature2idx)
        for i in range(dimred_dict[feature]['target_dim']):
            colname = dimred_dict[feature]['algorithm'] + ':' + feature + ':' + str(i)
            feature2idx[colname] = nb_cols_sofar + i
            idx2feature[nb_cols_sofar + 1] = colname
        dimred_dict[feature]['colindices'] = list(
            range(nb_cols_sofar, nb_cols_sofar + dimred_dict[feature]['target_dim']))
    return feature2idx, idx2feature


def convert_data_to_matrix_with_embeddings(data,
                                           feature2idx,
                                           qual_features,
                                           binary_features,
                                           quant_features,
                                           dimred_dict,
                                           labelcol,
                                           label2idx):
    reduced_qual_features = [feature for feature in qual_features if feature not in dimred_dict.keys()]
    X, Y = convert_data_to_matrix(data, feature2idx,
                                  reduced_qual_features, binary_features, quant_features,
                                  labelcol, label2idx)
    for feature in dimred_dict:
        featurevalX, _ = convert_data_to_matrix(data, dimred_dict[feature]['dictionary'],
                                                [feature], [], [],
                                                labelcol, label2idx)
        redfeaturevalX = dimred_dict[feature]['transformer'].transform(featurevalX.toarray())
        X[:, dimred_dict[feature]['colindices']] = redfeaturevalX
    return X, Y



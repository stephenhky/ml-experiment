import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.core import generate_columndict, convert_data_to_matrix


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


class NumericallyPreparedDataset(Dataset):
    def __init__(self,
                 data_iterator,
                 feature2idx,
                 qual_features, binary_features, quant_features,
                 dimred_dict,
                 labelcol, label2idx,
                 assigned_partitions=None, interested_partitions=[],
                 device='cpu'):
        super(NumericallyPreparedDataset, self).__init__()
        self.feature2idx = feature2idx
        self.qual_features = qual_features
        self.binary_features = binary_features
        self.quant_features = quant_features
        self.dimred_dict = dimred_dict
        self.labelcol = labelcol
        self.label2idx = label2idx
        self.device = torch.device(device)

        if assigned_partitions is None:
            filtered_data = [datum for datum in data_iterator]
        else:
            filtered_data = [datum
                             for datum, partition in zip(data_iterator, assigned_partitions)
                             if partition in interested_partitions]
        self.X, self.Y = convert_data_to_matrix_with_embeddings(filtered_data,
                                                                self.feature2idx,
                                                                self.qual_features,
                                                                self.binary_features,
                                                                self.quant_features,
                                                                self.dimred_dict,
                                                                self.labelcol,
                                                                self.label2idx)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return Tensor(self.X[idx, :]), Tensor(self.Y[idx, :])

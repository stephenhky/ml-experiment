
import os
from glob import glob
import tempfile

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.core import generate_columndict, convert_data_to_matrix
from ..data.dataload import iterate_json_files_directory
from ..data.adding_features import adding_no_features


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
        x = Tensor(self.X[idx, :])
        y = Tensor(self.Y[idx, :])
        return x, y


class CachedNumericallyPreparedDataset(Dataset):
    def __init__(self,
                 datadir,   # JSON format
                 batch_size,
                 feature2idx,
                 qual_features, binary_features, quant_features,
                 dimred_dict,
                 labelcol, label2idx,
                 assigned_partitions=None, interested_partitions=[],
                 h5dir=None,
                 filename_fmt='data_{0:09d}.h5',
                 device='cpu'):
        super(CachedNumericallyPreparedDataset, self).__init__()
        self.datadir = datadir
        self.batch_size = batch_size
        self.feature2idx = feature2idx
        self.qual_features = qual_features
        self.binary_features = binary_features
        self.quant_features = quant_features
        self.dimred_dict = dimred_dict
        self.labelcol = labelcol
        self.label2idx = label2idx
        self.assigned_partitions = assigned_partitions
        self.interested_partitions = interested_partitions
        self.filename_fmt = filename_fmt
        self.device = torch.device(device)

        # writing to h5 files
        nbdata = 0
        fileid = 0
        if h5dir is None:
            self.h5tempdir = tempfile.TemporaryDirectory()
            self.h5dir = self.h5tempdir.name
        else:
            self.h5dir = h5dir
        batch_data = []
        idx2feature = [None]*len(feature2idx)
        for col, i in feature2idx.items():
            idx2feature[i] = col
        idx2label = [None]*len(label2idx)
        for col, i in label2idx.items():
            idx2label[i] = col
        for i, datum in enumerate(iterate_json_files_directory(self.datadir,
                                                               feature_adder=adding_no_features)):
            if self.assigned_partitions is not None and not (self.assigned_partitions[i] in self.interested_partitions):
                continue
            batch_data.append(datum)
            nbdata += 1
            if nbdata % batch_size == 0:
                self.write_data_h5(batch_data, idx2feature, idx2label, self.filename_fmt.format(fileid))
                fileid += 1
                batch_data = []

        if len(batch_data) > 0:
            self.write_data_h5(batch_data, idx2feature, idx2label, self.filename_fmt.format(fileid))
        self.nbdata = nbdata
        self.nbfiles = fileid + 1

        self.nbinputs = len(self.feature2idx)
        self.nboutputs = len(self.label2idx)

        # cached
        self.current_fileid = -1

    def write_data_h5(self, batch_data, xcolumns, ycolumns, filename):
        X, Y = convert_data_to_matrix_with_embeddings(batch_data,
                                                      self.feature2idx,
                                                      self.qual_features,
                                                      self.binary_features,
                                                      self.quant_features,
                                                      self.dimred_dict,
                                                      self.labelcol,
                                                      self.label2idx)
        df = pd.DataFrame(X.toarray(), columns=xcolumns)
        for i in range(Y.shape[1]):
            df[ycolumns[i]] = Y.toarray()[:, i]
        df.to_hdf(os.path.join(self.h5dir, filename), key=os.path.basename(filename)[:-3])

    def __len__(self):
        return self.nbdata

    def __getitem__(self, idx):
        fileid = idx // self.batch_size
        pos = idx % self.batch_size
        if self.current_fileid != fileid:
            self.df = pd.read_hdf(os.path.join(self.h5dir, self.filename_fmt.format(fileid)))
        return Tensor(self.df.iloc[pos, :self.nbinputs]), Tensor(self.df.iloc[pos, -self.nboutputs:])

    def get_batch(self, fileid):
        self.current_fileid = fileid
        self.df = pd.read_hdf(os.path.join(self.h5dir, self.filename_fmt.format(fileid)))
        return Tensor(np.array(self.df.iloc[:, :self.nbinputs])), Tensor(np.array(self.df.iloc[:, -self.nboutputs:]))

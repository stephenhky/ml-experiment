import math
import os
import tempfile
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..data.adding_features import adding_no_features
from ..data.dataload import iterate_json_files_directory

from .datatransform import convert_data_to_matrix_with_embeddings


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
                 h5dir,
                 batch_size,
                 feature2idx,
                 qual_features, binary_features, quant_features,
                 dimred_dict,
                 labelcol, label2idx,
                 assigned_partitions=None, interested_partitions=[],
                 filename_fmt='data_{0:09d}.h5',
                 device='cpu'):
        super(CachedNumericallyPreparedDataset, self).__init__()
        self.store_parameter(h5dir,
                 batch_size,
                 feature2idx,
                 qual_features, binary_features, quant_features,
                 dimred_dict,
                 labelcol, label2idx,
                 assigned_partitions, interested_partitions,
                 filename_fmt,
                 device)
        self.reshuffle_batch = (self.assigned_partitions is not None)
        self.wrangle_batch()

    def store_parameter(self, h5dir,
                 batch_size,
                 feature2idx,
                 qual_features, binary_features, quant_features,
                 dimred_dict,
                 labelcol, label2idx,
                 assigned_partitions, interested_partitions,
                 filename_fmt,
                 device):
        self.h5dir = h5dir
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

        # calculation
        self.nbinputs = len(self.feature2idx)
        self.nboutputs = len(self.label2idx)

        # cached
        self.nbfiles = len(glob(os.path.join(self.h5dir, '*.h5')))
        self.current_fileid = -1

    def wrangle_batch(self):
        if not self.reshuffle_batch:
            self.dataidx = np.arange(len(self.assigned_partitions))
            self.nbbatches = self.nbfiles
        else:
            self.dataidx = [self.assigned_partitions[i]
                            for i in range(len(self.assigned_partitions))
                            if self.assigned_partitions[i] in self.interested_partitions]
            self.dataidx = np.array(self.dataidx)
            self.nbbatches = math.ceil(len(self.dataidx) / self.batch_size)
        self.data_fileids, self.data_filepos = self.calculate_fileid_pos(self.dataidx)

    def calculate_fileid_pos(self, idx):
        fileid = idx // self.batch_size
        pos = idx % self.batch_size
        return fileid, pos

    def __len__(self):
        return len(self.dataidx)

    def __getitem__(self, idx):
        fileid, pos = self.calculate_fileid_pos(idx)
        if self.current_fileid != fileid:
            self.df = pd.read_hdf(os.path.join(self.h5dir, self.filename_fmt.format(fileid)))
        return Tensor(self.df.iloc[pos, :self.nbinputs]), Tensor(self.df.iloc[pos, -self.nboutputs:])

    def get_batch(self, batchid):
        if not self.reshuffle_batch:
            self.current_fileid = batchid
            self.df = pd.read_hdf(os.path.join(self.h5dir, self.filename_fmt.format(batchid)))
            return Tensor(np.array(self.df.iloc[:, :self.nbinputs])), Tensor(np.array(self.df.iloc[:, -self.nboutputs:]))
        else:
            fileids = self.data_fileids[batchid*self.batch_size:(batchid+1)*self.batch_size]
            filepos = self.data_filepos[batchid*self.batch_size:(batchid+1)*self.batch_size]
            unique_fileids = set(fileids)
            x_to_return = None
            y_to_return = None
            for fileid in unique_fileids:
                if fileid != self.current_fileid:
                    self.df = pd.read_hdf(os.path.join(self.h5dir, self.filename_fmt.format(batchid)))
                    self.current_fileid = fileid
                pos = filepos[ fileids == fileid]
                newx = np.array(self.df.iloc[pos, :self.nbinputs])
                newy = np.array(self.df.iloc[pos, -self.nboutputs:])
                x_to_return = newx if x_to_return is None else np.append(x_to_return, newx, axis=0)
                y_to_return = newy if y_to_return is None else np.append(y_to_return, newy, axis=0)
            return Tensor(x_to_return), Tensor(y_to_return)


class PreparingCachedNumericallyPreparedDataset(CachedNumericallyPreparedDataset):
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
        Dataset.__init__(self)
        if h5dir is None:
            self.h5tempdir = tempfile.TemporaryDirectory()    # storing the context so that it is not removed after exiting the constructor
            h5dir = self.h5tempdir.name
        self.store_parameter(h5dir,
                 batch_size,
                 feature2idx,
                 qual_features, binary_features, quant_features,
                 dimred_dict,
                 labelcol, label2idx,
                 assigned_partitions, interested_partitions,
                 filename_fmt,
                 device)
        self.reshuffle_batch = False
        self.datadir = datadir
        self.prepare_h5_files()

    def prepare_h5_files(self):
        # writing to h5 files
        nbdata = 0
        fileid = 0

        batch_data = []
        idx2feature = [None]*len(self.feature2idx)
        for col, i in self.feature2idx.items():
            idx2feature[i] = col
        idx2label = [None]*len(self.label2idx)
        for col, i in self.label2idx.items():
            idx2label[i] = col
        for i, datum in enumerate(iterate_json_files_directory(self.datadir,
                                                               feature_adder=adding_no_features)):
            if self.assigned_partitions is not None and not (self.assigned_partitions[i] in self.interested_partitions):
                continue
            batch_data.append(datum)
            nbdata += 1
            if nbdata % self.batch_size == 0:
                self.write_data_h5(batch_data, idx2feature, idx2label,
                                   self.filename_fmt.format(fileid))
                fileid += 1
                batch_data = []

        if len(batch_data) > 0:
            self.write_data_h5(batch_data, idx2feature, idx2label,
                               self.filename_fmt.format(fileid))
        self.nbdata = nbdata
        self.nbfiles = fileid + 1
        self.nbbatches = self.nbfiles

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
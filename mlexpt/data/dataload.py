
import os
import tempfile
from glob import glob
import json
from collections import OrderedDict

import numpy as np

from .adding_features import adding_no_features


def iterate_json_data(filepath,
                      columns_to_keep=None,
                      feature_adder=adding_no_features,
                      data_filter=lambda datum: True,
                      missing_val_default={}):
    inputfile = open(filepath, 'r')
    for line in inputfile:
        datum = json.loads(line)
        datum = feature_adder(datum)
        if not data_filter(datum):
            continue
        if columns_to_keep is not None:
            filtered_datum = OrderedDict()
            for column in columns_to_keep:
                filtered_datum[column] = datum[column]
                if column in missing_val_default.keys() and datum[column] is None:
                    filtered_datum[column] = missing_val_default[column]
            yield filtered_datum
        else:
            yield OrderedDict(datum)


def iterate_json_files_directory(dir,
                                 columns_to_keep=None,
                                 feature_adder=adding_no_features,
                                 data_filter=lambda datum: True,
                                 missing_val_default={}
                                 ):
    print('\tReading {}'.format(dir))
    print('\tColumns: {}'.format(', '.join(columns_to_keep) if columns_to_keep is not None else 'ALL'))
    for filepath in glob(os.path.join(dir, '*.json')):
        for datum in iterate_json_data(filepath,
                                       columns_to_keep=columns_to_keep,
                                       feature_adder=feature_adder,
                                       data_filter=data_filter,
                                       missing_val_default=missing_val_default):
            yield datum


def process_data(traindatafilepath, qual_features, binary_features, quant_features,
                 target_label,
                 feature_adder=adding_no_features,
                 nb_lines_per_tempfile=10000,
                 data_filter=lambda datum: True,
                 missing_val_default={},
                 filename_fmt='data_{0:09d}.json'):
    tempdir = tempfile.TemporaryDirectory()
    fileid = 0
    tmpfile = None
    nbdata = 0
    for i, datum in enumerate(iterate_json_data(traindatafilepath,
                                                columns_to_keep=qual_features+binary_features+quant_features+[target_label],
                                                feature_adder=feature_adder,
                                                data_filter=data_filter,
                                                missing_val_default=missing_val_default)):
        if i % nb_lines_per_tempfile == 0:
            if tmpfile is not None:
                tmpfile.close()
            tmpfile = open(os.path.join(tempdir.name, filename_fmt.format(fileid)), 'w')
            fileid += 1
            print('\tRead {} lines...'.format(i))
        nbdata += 1
        tmpfile.write(json.dumps(datum)+'\n')
    tmpfile.close()
    return tempdir, nbdata


def assign_partitions(nbdata, cv_nfold, heldout_fraction, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice([-1] + list(range(cv_nfold)),  # -1 indicating hold-out set
                            p=[heldout_fraction] + [(1 - heldout_fraction) / cv_nfold] * cv_nfold,
                            size=nbdata)




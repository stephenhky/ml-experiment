
import json
import os
from warnings import warn

import joblib
import numpy as np
from torch.utils.data import DataLoader

from .data.adding_features import adding_no_features
from .ml.models import classifiers_dict
from .utils.caching import CachedNumericallyPreparedDataset
from .utils.datatransform import convert_data_to_matrix_with_embeddings


def persist_model_files(dirpath, model, dimred_dict, feature2idx, label2idx, config):
    if not os.path.exists(dirpath):
        warn('Directory {} does not exist, but is being created...'.format(dirpath))
        os.makedirs(dirpath)
    if not os.path.isdir(dirpath):
        raise IOError('Path {} is not a directory!'.format(dirpath))

    # save all dicts into metadata JSONs
    metadata = {
        'model': {
            key: val
            for key, val in config['model'].items()
            if key in ['qualitative_features',
                       'binary_features',
                       'quantitative_features',
                       'target',
                       'algorithm']
                  },
        'dimred_dict': {
            feature: {
                key: val
                for key, val in dimred_dict[feature].items()
                if key in ['dictionary', 'target_dim', 'algorithm', 'colindices']
            }
            for feature in dimred_dict.keys()
        },
        'feature2idx': feature2idx,
        'label2idx': label2idx
    }

    # saving the model
    model.persist(open(os.path.join(dirpath, 'modelobj.pkl'), 'wb'))
    # saving the information about encodings
    for feature in dimred_dict:
        transformer = dimred_dict[feature]['transformer']
        transformer.trim()
        transformer_modelfilename = feature + '_'+dimred_dict[feature]['algorithm'] + \
                                        '_{}.pkl'.format(dimred_dict[feature]['target_dim'])
        transformer_modelpath = os.path.join(dirpath, transformer_modelfilename)
        metadata['dimred_dict'][feature]['transformer_model_filename'] = transformer_modelfilename
        transformer.persist(transformer_modelpath)

    # saving metadata
    json.dump(metadata, open(os.path.join(dirpath, 'metadata.json'), 'w'))


def model_predict_proba(model, qual_features, binary_features, quant_features,
                        dimred_dict, feature2idx, testdata):
    if not isinstance(testdata, list):
        testdata = [testdata]

    X, _ = convert_data_to_matrix_with_embeddings(testdata, feature2idx,
                                                  qual_features, binary_features, quant_features,
                                                  dimred_dict, None, {})
    pred_Y = model.predict_proba(X.toarray())
    return pred_Y


def model_predict_on_cached_dataset(
        model, h5dir, batch_size,
        feature2idx, qual_features, binary_features, quant_features,
        dimred_dict, labelcol, label2idx,
        assigned_partitions, interested_partitions,
        device='cpu'):
    test_dataset = CachedNumericallyPreparedDataset(h5dir,
                                                    batch_size,
                                                    feature2idx,
                                                    qual_features, binary_features, quant_features,
                                                    dimred_dict, labelcol, label2idx,
                                                    assigned_partitions=assigned_partitions,
                                                    interested_partitions=interested_partitions,
                                                    device=device)
    if len(test_dataset) > 0:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        predicted_Y = None
        test_Y = None
        for data in test_dataloader:
            x, test_y = data
            new_pred_y = model.predict_proba(x)
            if predicted_Y is None:
                predicted_Y = new_pred_y
            else:
                predicted_Y = np.append(predicted_Y, new_pred_y, axis=0)
            if test_Y is None:
                test_Y = np.array(test_y)
            else:
                test_Y = np.append(test_Y, np.array(test_y), axis=0)
        return predicted_Y, test_Y
    else:
        return np.array([]), np.array([])


def train_model(h5dir, batch_size, feature2idx,
                qual_features, binary_features, quant_features,
                dimred_dict, labelcol, label2idx,
                assigned_partitions, interested_partitions,
                model_class, model_param, device='cpu'):
    train_dataset = CachedNumericallyPreparedDataset(h5dir,
                                                     batch_size,
                                                     feature2idx,
                                                     qual_features, binary_features, quant_features,
                                                     dimred_dict, labelcol, label2idx,
                                                     assigned_partitions=assigned_partitions,
                                                     interested_partitions=interested_partitions,
                                                     device=device)
    model = model_class(**model_param)
    model.fit_batch(train_dataset)
    return model


class CompactExperimentalModel:
    def __init__(self, modeldir, feature_adder=adding_no_features, modelclass=None, modelloadkwargs={}):
        self.modeldir = modeldir
        self.feature_adder = feature_adder
        self.metadata = json.load(open(os.path.join(modeldir, 'metadata.json'), 'r'))

        if self.metadata['model']['algorithm'] in classifiers_dict:
            algorithm = self.metadata['model']['algorithm']
            modelpath = os.path.join(modeldir, 'modelobj.pkl')
            self.model = classifiers_dict[algorithm].load(modelpath, **modelloadkwargs)
        elif modelclass is not None:
            modelpath = os.path.join(modeldir, 'modelobj.pkl')
            self.model = modelclass.load(modelpath, **modelloadkwargs)
        else:
            raise Exception('Invalid Model!')

        for feature in self.metadata['dimred_dict']:
            transformer = joblib.load(os.path.join(modeldir,
                                                   self.metadata['dimred_dict'][feature]['transformer_model_filename']))
            self.metadata['dimred_dict'][feature]['transformer'] = transformer

    def predict_proba(self, testdata):
        return model_predict_proba(self.model,
                                   self.metadata['model']['qualitative_features'],
                                   self.metadata['model']['binary_features'],
                                   self.metadata['model']['quantitative_features'],
                                   self.metadata['dimred_dict'],
                                   self.metadata['feature2idx'],
                                   testdata)

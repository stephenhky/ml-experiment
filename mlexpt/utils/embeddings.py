
import pickle
import tempfile
from warnings import warn

from .core import generate_columndict, convert_data_to_matrix
from ..data.dataload import iterate_json_files_directory
from ..ml.encoders.dictembedding import DictEmbedding
from ..ml.models import encoders_dict
from .caching import PreparingCachedNumericallyPreparedDataset


def embed_features(dr_config, alldata):
    dimred_dict = {}
    for feature in dr_config:
        print("\t Embedding Feature: {} ({})".format(feature, dr_config[feature]['algorithm']))
        featureval2idx, idx2featureval = generate_columndict(alldata, [feature], [], [])
        featurevalX, _ = convert_data_to_matrix(alldata,
                                                featureval2idx, [feature], [], [],
                                                None, [])

        if dr_config[feature].get('transformer_class') is not None:
            param = dr_config[feature].get('transform_param', {})
            transformer = dr_config[feature]['transformer_class'](**param)
        elif dr_config[feature]['algorithm'] == 'embedding_dict':
            transformer = DictEmbedding(featureval2idx,
                                        pickle.load(open(dr_config[feature]['filepath'], 'rb'))
                                        )
            dr_config[feature]['target_dim'] = transformer.target_dim
        elif dr_config[feature]['algorithm'] in encoders_dict.keys():
            transformer = encoders_dict[dr_config[feature]['algorithm']](n_components=dr_config[feature]['target_dim'])
        else:
            warn('Encoder {} is not configured.'.format(dr_config[feature]['algorithm']))
            continue

        transformer.fit(featurevalX.toarray())
        dimred_dict[feature] = {'transformer': transformer,
                                'dictionary': featureval2idx,
                                'target_dim': dr_config[feature]['target_dim'],
                                'algorithm': dr_config[feature].get('algorithm', '')}

    return dimred_dict


def embed_features_cacheddataset(dr_config, datadir, batch_size=10000):
    dimred_dict = {}
    for feature in dr_config:
        print("\t Embedding Feature: {} ({})".format(feature, dr_config[feature]['algorithm']))
        featureval2idx, idx2featureval = generate_columndict(iterate_json_files_directory(datadir,
                                                                                          columns_to_keep=[feature]),
                                                             [feature], [], [])

        if dr_config[feature].get('transformer_class') is not None:
            param = dr_config[feature].get('transform_param', {})
            transformer = dr_config[feature]['transformer_class'](**param)
        elif dr_config[feature]['algorithm'] == 'embedding_dict':
            transformer = DictEmbedding(featureval2idx,
                                        pickle.load(open(dr_config[feature]['filepath'], 'rb'))
                                        )
            dr_config[feature]['target_dim'] = transformer.target_dim
        elif dr_config[feature]['algorithm'] in encoders_dict.keys():
            transformer = encoders_dict[dr_config[feature]['algorithm']](n_components=dr_config[feature]['target_dim'])
        else:
            warn('Encoder {} is not configured.'.format(dr_config[feature]['algorithm']))
            continue

        h5datatempdir = tempfile.TemporaryDirectory()
        dataset = PreparingCachedNumericallyPreparedDataset(datadir,
                                                            batch_size,
                                                            featureval2idx,
                                                            [feature],
                                                            [],
                                                            [],
                                                            dimred_dict,
                                                            None,
                                                            {},
                                                            h5dir=h5datatempdir.name)
        transformer.fit_batch(dataset)
        dimred_dict[feature] = {'transformer': transformer,
                                'dictionary': featureval2idx,
                                'target_dim': dr_config[feature]['target_dim'],
                                'algorithm': dr_config[feature].get('algorithm', '')}

    return dimred_dict

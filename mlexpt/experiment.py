from functools import partial
from time import time
import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .data.adding_features import adding_no_features
from .data.dataload import process_data, iterate_json_files_directory, assign_partitions
from .metrics.statistics import extracting_stats_run, compute_average_overall_performance, \
    compute_average_performances_per_class, count_occurences_of_feature_values
from .ml.models import classifiers_dict
from .utils.core import generate_columndict
from .utils.embeddings import embed_features_cacheddataset
from .utils.datatransform import generate_columndict_withembeddings
from .utils.caching import CachedNumericallyPreparedDataset, PreparingCachedNumericallyPreparedDataset
from .modelrunio import persist_model_files


NB_LINES_PER_TEMPFILE = 500
BATCH_SIZE = 500

def add_multiple_features(add_feature_functions):
    def returned_function(datum, add_feature_functions):
        for function in add_feature_functions:
            function(datum)
        return datum
    return partial(returned_function, add_feature_functions=add_feature_functions)


def run_experiment(config,
                   feature_adder=adding_no_features,
                   nb_lines_per_tempfile=NB_LINES_PER_TEMPFILE,
                   data_filter=lambda datum: True,
                   model_class=None,
                   batch_size=BATCH_SIZE):
    ## model config
    qual_features = config['model'].get('qualitative_features', [])
    binary_features = config['model'].get('binary_features', [])
    quant_features = config['model'].get('quantitative_features', [])
    dr_config = config['model'].get('embedding_schemes', {})
    labelcol = config['model']['target']
    algorithm = config['model']['algorithm']
    model_param = config['model']['model_parameters']
    ## cross validation setup
    do_cv = config['train']['cross_validation']
    cv_nfold = config['train'].get('cv_nfold', 5)
    heldout_fraction = config['train']['heldout_fraction']
    to_persist_model = config['train']['persist_model']
    final_model_path = config['train']['model_path']
    # data
    datapath = config['data']['path']
    missing_val_default = config['data']['missing_value_filling']
    data_device = config['data']['torchdevice']
    h5dir = config['data'].get('h5dir', './.h5')
    # statistics
    topN = config['statistics']['topN']
    to_compute_class_performances = config['statistics'].get('compute_class_performance', False)
    class_performance_excel_file = config['statistics'].get('class_performances_excel', None)

    # starting timer
    starttime = time()

    # loading data, adding features, trimming the data
    print('Reading data....')
    tempdir, nbdata = process_data(datapath,
                                   qual_features, binary_features, quant_features,
                                   labelcol,
                                   feature_adder=feature_adder,
                                   nb_lines_per_tempfile=nb_lines_per_tempfile,
                                   data_filter=data_filter,
                                   missing_val_default=missing_val_default)
    print("Temporary directory: {}".format(tempdir.name))
    print('Number of data: {}'.format(nbdata))
    print(' ')
    data_processing_endtime = time()

    # getting labels
    print('Generating target labels...')
    label2idx, idx2label = generate_columndict(iterate_json_files_directory(tempdir.name),
                                              [labelcol], [], [])
    target_label_dict = {key[len(labelcol)+1:]: value for key, value in label2idx.items()}
    print('\tNumber of labels: {}'.format(len(label2idx)))
    count_dict = count_occurences_of_feature_values(iterate_json_files_directory(tempdir.name), labelcol)
    for label, count in sorted(count_dict.items(), key=lambda item: item[1], reverse=True):
        print('{}: {}'.format(label, count))
    print(' ')

    # dimensionality reduction of embedding
    print('Embedding')
    dimred_dict = embed_features_cacheddataset(dr_config, tempdir.name, batch_size=BATCH_SIZE)

    # generating columndict with dimensionality reduction or embedding
    print('Generating columns dictionary...')
    feature2idx, idx2feature = generate_columndict_withembeddings(iterate_json_files_directory(tempdir.name),
                                                                  qual_features,
                                                                  binary_features,
                                                                  quant_features,
                                                                  dimred_dict)
    columndict_generation_endtime = time()

    # partition assignment
    # important: even if cross-validation will not be performed
    partitions = assign_partitions(nbdata, cv_nfold, heldout_fraction)

    # making numerical transform
    if not os.path.exists(h5dir) or not os.path.isdir(h5dir):
        os.makedirs(h5dir)
    print('Numerically transformed files stored in: {}'.format(h5dir))
    _ = PreparingCachedNumericallyPreparedDataset(tempdir.name,
                                                  batch_size,
                                                  feature2idx,
                                                  qual_features, binary_features, quant_features,
                                                  dimred_dict, labelcol, label2idx,
                                                  assigned_partitions=partitions,
                                                  interested_partitions=list(set(partitions)),
                                                  h5dir=h5dir,
                                                  device=data_device)
    alldataset_h5transform_endtime = time()

    # cross-validation
    overall_performances = []
    top_results_by_class = []
    weighted_results_by_class = []
    hit_results_by_class = []
    if do_cv:
        print('Cross Validation')

        for cv_round in range(cv_nfold):
            # train
            print('Round {}'.format(cv_round))
            train_dataset = CachedNumericallyPreparedDataset(h5dir,
                                                             batch_size,
                                                             feature2idx,
                                                             qual_features, binary_features, quant_features,
                                                             dimred_dict, labelcol, label2idx,
                                                             assigned_partitions=partitions,
                                                             interested_partitions=[partition
                                                                                    for partition in range(cv_nfold)
                                                                                    if partition != cv_round],
                                                             device=data_device)

            if model_class is None:
                model = classifiers_dict[algorithm](**model_param)
            else:
                model = model_class(**model_param)
            model.fit_batch(train_dataset)

            # test
            test_dataset = CachedNumericallyPreparedDataset(h5dir,
                                                            batch_size,
                                                            feature2idx,
                                                            qual_features, binary_features, quant_features,
                                                            dimred_dict, labelcol, label2idx,
                                                            assigned_partitions=partitions,
                                                            interested_partitions=[cv_round],
                                                            device=data_device)
            predicted_Y = model.predict_proba_batch(test_dataset)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
            test_Y = None
            for data in test_dataloader:
                _, test_y = data
                if test_Y is None:
                    test_Y = np.array(test_y)
                else:
                    test_Y = np.append(test_Y, np.array(test_y), axis=0)

            # statistics
            overall_performance, top_result_by_class, weighted_result_by_class, hit_result_by_class = \
                extracting_stats_run(predicted_Y, test_Y, target_label_dict, topN)
            overall_performances.append(overall_performance)
            top_results_by_class.append(top_result_by_class)
            weighted_results_by_class.append(weighted_result_by_class)
            hit_results_by_class.append(hit_result_by_class)

            print(overall_performance)

    cross_validation_endtime = time()

    # train a final model
    if to_persist_model:
        print('Training final model...')
        dataset = CachedNumericallyPreparedDataset(h5dir,
                                                   batch_size,
                                                   feature2idx,
                                                   qual_features, binary_features, quant_features,
                                                   dimred_dict, labelcol, label2idx,
                                                   assigned_partitions=partitions,
                                                   interested_partitions=[partition
                                                                          for partition in range(cv_nfold)
                                                                          if partition >= 0],
                                                   device=data_device)
        if model_class is None:
            model = classifiers_dict[algorithm](**model_param)
        else:
            model = model_class(**model_param)
        model.fit_batch(dataset)
        print('Saving the final model...')
        persist_model_files(final_model_path, model, dimred_dict, feature2idx, label2idx, config)

        print('Testing the final model...')
        heldout_dataset = CachedNumericallyPreparedDataset(h5dir,
                                                           batch_size,
                                                           feature2idx,
                                                           qual_features, binary_features, quant_features,
                                                           dimred_dict, labelcol, label2idx,
                                                           assigned_partitions=partitions,
                                                           interested_partitions=[-1],
                                                           device=data_device)
        if len(heldout_dataset) > 0:
            heldout_dataloader = DataLoader(heldout_dataset, batch_size=batch_size)
            predicted_Y = None
            test_Y = None
            for data in heldout_dataloader:
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

            final_model_overall_performance, _, _, _ = \
                extracting_stats_run(predicted_Y, test_Y, target_label_dict, topN)

    finalmodel_training_endtime = time()

    # output statistics
    print('Total time: {0:.1f} sec'.format(finalmodel_training_endtime-starttime))
    print('\tData processing time: {0:.1f} sec'.format(data_processing_endtime-starttime))
    print('\tColumn dictionary generation time: {0:.1f} sec'.format(columndict_generation_endtime-data_processing_endtime))
    print('\tNumerical transformation time: {0:.1f} sec'.format(alldataset_h5transform_endtime-columndict_generation_endtime))
    print('\tCross validation time: {0:.1f} sec'.format(cross_validation_endtime-alldataset_h5transform_endtime))
    print('\tFinal model training time: {0:.1f} sec'.format(finalmodel_training_endtime-cross_validation_endtime))

    if do_cv:
        # print overall performances
        average_overall_performance = compute_average_overall_performance(overall_performances)
        print('Final Measurement')
        print('=================')
        print('Top accuracy: {0:.2f}%'.format(average_overall_performance['top1_accuracy']*100))
        print('Weighted accuracy: {0:.2f}%'.format(average_overall_performance['weighted_accuracy']*100))
        print('Hit accuracy: {0:.2f}%'.format(average_overall_performance['hit_accuracy']*100))

        # each class output
        if to_compute_class_performances:
            average_top_results_by_class = compute_average_performances_per_class(top_results_by_class,
                                                                                  target_label_dict.keys())
            average_weighted_results_by_class = compute_average_performances_per_class(weighted_results_by_class,
                                                                                       target_label_dict.keys())
            average_hit_result_by_class = compute_average_performances_per_class(hit_results_by_class,
                                                                                 target_label_dict.keys())

            average_top_results_by_class_df = pd.DataFrame.from_dict(average_top_results_by_class, orient='index')
            print('Top Results by Class')
            print('====================')
            print(average_top_results_by_class_df)
            average_weighted_results_by_class_df = pd.DataFrame.from_dict(average_weighted_results_by_class, orient='index')
            print('Weighted Results by Class')
            print('=========================')
            print(average_weighted_results_by_class_df)
            average_hit_result_by_class_df = pd.DataFrame.from_dict(average_hit_result_by_class, orient='index')
            print('Hit Results by Class')
            print('====================')
            print(average_hit_result_by_class_df)
            if class_performance_excel_file is not None:
                excelWriter = pd.ExcelWriter(class_performance_excel_file)
                average_top_results_by_class_df.to_excel(excel_writer=excelWriter, sheet_name='Top Results')
                average_weighted_results_by_class_df.to_excel(excel_writer=excelWriter, sheet_name='Weighted Results')
                average_hit_result_by_class_df.to_excel(excel_writer=excelWriter, sheet_name='Hit Results')
                excelWriter.close()

    if to_persist_model and len(heldout_dataset) > 0:
        print('Held-out Measurement')
        print('=================')
        print('Top accuracy: {0:.2f}%'.format(final_model_overall_performance['top1_accuracy']*100))
        print('Weighted accuracy: {0:.2f}%'.format(final_model_overall_performance['weighted_accuracy']*100))
        print('Hit accuracy: {0:.2f}%'.format(final_model_overall_performance['hit_accuracy']*100))

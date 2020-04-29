
from collections import defaultdict

import numpy as np


def count_matches(modeloutputY, testY, topN):
    assert modeloutputY.shape[0] == testY.shape[0]
    nb_testdata = modeloutputY.shape[0]
    nbclasses = testY.shape[1]

    #getting labels for test data
    testY_labelindices = np.argmax(testY if isinstance(testY, np.ndarray) else testY.toarray(), axis=1)
    # top1 accuracy
    modeloutputY_labelindices = np.argmax(modeloutputY, axis=1)
    nb_equals = np.sum(testY_labelindices==modeloutputY_labelindices)
    #topN weighted accuracy
    sorted_index_matrix = np.argsort(modeloutputY, axis=1)
    nb_weighted_equals = nb_equals
    nb_hits= nb_equals
    for i in range(1, topN):
        nb_weighted_equals += np.sum(testY_labelindices == sorted_index_matrix[:, nbclasses - i - 1]) * (topN - i) / topN
        nb_hits += np.sum(testY_labelindices == sorted_index_matrix[:, nbclasses - i - 1])

    return nb_equals, nb_weighted_equals, nb_hits


def compute_confusion_matrices(modeloutputY, testY, topN):
    assert modeloutputY.shape[1] == testY.shape[1]
    nbclasses = testY.shape[1]

    # getting labels for test data
    modeloutputY_labelindices = np.argmax(modeloutputY, axis=1)
    # top1 accuracy
    testY_labelindices = np.argmax(testY if isinstance(testY, np.ndarray) else testY.toarray(), axis=1)
    # topN weighted accuracy
    sorted_index_matrix = np.argsort(modeloutputY, axis=1)

    top_confusion_matrix = np.zeros((nbclasses, nbclasses))
    weighted_confusion_matrix = np.zeros((nbclasses, nbclasses))
    hit_confusion_matrix = np.zeros((nbclasses, nbclasses))
    for predicted_label, real_label in zip(modeloutputY_labelindices, testY_labelindices):
        top_confusion_matrix[predicted_label, real_label] += 1
        weighted_confusion_matrix[predicted_label, real_label] += 1
        hit_confusion_matrix[predicted_label, real_label] += 1
    for i in range(1, topN):
        for predicted_i_label, real_label in zip(sorted_index_matrix[:, nbclasses - i - 1], testY_labelindices):
            weighted_confusion_matrix[predicted_i_label, real_label] += (topN - i) / topN
            hit_confusion_matrix[predicted_i_label, real_label] += 1

    return top_confusion_matrix, weighted_confusion_matrix, hit_confusion_matrix


def extracting_stats_run(modeloutputY, testY, label2idx, topN):
    nb_testdata = testY.shape[0]

    nb_equals, nb_weighted_equals, nb_hits = count_matches(modeloutputY, testY, topN)
    top1_accuracy = nb_equals / nb_testdata
    weighted_accuracy= nb_weighted_equals / nb_testdata
    hit_accuracy = nb_hits / nb_testdata
    overall_performance = {'nb_testdata': nb_testdata,
                           'nb_equals': nb_equals,
                           'nb_weighted_equals': nb_weighted_equals,
                           'nb_hits': nb_hits,
                           'top1_accuracy': top1_accuracy,
                           'weighted_accuracy': weighted_accuracy,
                           'hit_accuracy': hit_accuracy,
                           'topN': topN
                           }

    # calculating the confusion matrix
    top_confusion_matrix, weighted_confusion_matrix, hit_confusion_matrix = \
        compute_confusion_matrices(modeloutputY, testY, topN)
    # calculating metrics
    top_result_by_class = {}
    weighted_result_by_class = {}
    hit_result_by_class = {}
    for label, idx in label2idx.items():
        # note: the definitions are different
        top_recall = top_confusion_matrix[idx, idx] / np.sum(top_confusion_matrix[:, idx])
        top_precision = top_confusion_matrix[idx, idx] / np.sum(top_confusion_matrix[idx, :])
        top_f_score = 2*top_recall*top_precision / (top_recall + top_precision)
        top_result_by_class[label] = {'recall': top_recall,
                                      'precision': top_precision,
                                      'f_score': top_f_score}
        weighted_recall = weighted_confusion_matrix[idx, idx] / np.sum(top_confusion_matrix[:, idx])
        weighted_precision = weighted_confusion_matrix[idx, idx] / np.sum(weighted_confusion_matrix[idx, :])
        weighted_f_score = 2*weighted_recall*weighted_precision / (weighted_recall + weighted_precision)
        weighted_result_by_class[label] = {'recall': weighted_recall,
                                           'precision': weighted_precision,
                                           'f_score': weighted_f_score}
        hit_recall = hit_confusion_matrix[idx, idx] / np.sum(top_confusion_matrix[:, idx])
        hit_precision = hit_confusion_matrix[idx, idx] / np.sum(weighted_confusion_matrix[idx, :])
        hit_f_score = 2*hit_recall*hit_precision / (hit_recall + hit_precision)
        hit_result_by_class[label] = {'recall': hit_recall,
                                      'precision': hit_precision,
                                      'f_score': hit_f_score}

    return overall_performance, top_result_by_class, weighted_result_by_class, hit_result_by_class


def compute_average_overall_performance(overall_performances):
    average_overall_performance = {'top1_accuracy': np.mean([performance['top1_accuracy']
                                                             for performance in overall_performances]),
                                   'weighted_accuracy': np.mean([performance['weighted_accuracy']
                                                                 for performance in overall_performances]),
                                   'hit_accuracy': np.mean([performance['hit_accuracy']
                                                            for performance in overall_performances])}
    return average_overall_performance


def compute_average_performances_per_class(class_performances, labels):
    average_performances_by_class = {}
    for label in labels:
        average_performances_by_class[label] = {
            'recall': np.mean([result[label]['recall'] for result in class_performances]),
            'precision': np.mean([result[label]['precision'] for result in class_performances]),
            'f_score': np.mean([result[label]['f_score'] for result in class_performances])
        }
    return average_performances_by_class


def count_occurences_of_feature_values(data_iterable, qual_feature):
    count_dict = defaultdict(lambda : 0)
    for datum in data_iterable:
        featureval = datum[qual_feature]
        if isinstance(featureval, list):
            for val in featureval:
                count_dict[val] += 1
        else:
            count_dict[featureval] += 1
    return dict(count_dict)

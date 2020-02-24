"""This module includes utilities to run cross-validation on general supervised learning methods."""
from __future__ import division
import numpy as np


def cross_validate(trainer, predictor, all_data, all_labels, folds, params):
    """Perform cross validation with random splits.

    :param trainer: function that trains a model from data with the template
             model = function(all_data, all_labels, params)
    :type trainer: function
    :param predictor: function that predicts a label from a single data point
                label = function(data, model)
    :type predictor: function
    :param all_data: d x n data matrix
    :type all_data: numpy ndarray
    :param all_labels: n x 1 label vector
    :type all_labels: numpy array
    :param folds: number of folds to run of validation
    :type folds: int
    :param params: auxiliary variables for training algorithm (e.g., regularization parameters)
    :type params: dict
    :return: tuple containing the average score and the learned models from each fold
    :rtype: tuple
    """
    scores = np.zeros(folds)

    d, n = all_data.shape

    indices = np.array(range(n), dtype=int)

    # pad indices to make it divide evenly by folds
    examples_per_fold = int(np.ceil(n / folds))
    ideal_length = int(examples_per_fold * folds)
    # use -1 as an indicator of an invalid index
    indices = np.append(indices, -np.ones(ideal_length - indices.size, dtype=int))
    assert indices.size == ideal_length

    indices = indices.reshape((examples_per_fold, folds))
    p = np.random.permutation(n)
    data_s,label_s = all_data[:,p],all_labels[p]
    labels = np.empty(folds)
    models = []

    for i in range(folds):
        indices[:,i] = data_s[:,0:examples_per_fold]
        labels[i] = label_s[0:examples_per_fold]

    for i in range(folds):
        c,p = np.delete(indices,i,1),indices[:,i]
        train = np.concatenate(c,1)
        model = trainer(train,labels[i],{})
        models.append(model)
        predict = predictor(p,model)
        scores[i] = np.mean(predict)

    score = np.mean(scores)

    return score, models
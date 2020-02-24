"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. (Optional. Can be empty)
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """

    labels = np.unique(train_labels)

    d = train_data.shape[0]
    num_classes = labels.size

    model = {
        "Priors": np.empty(num_classes),
        "Probs": np.empty(shape=(d,num_classes))
    }
    for i,l in enumerate(labels):
        array = train_labels == l
        count = np.count_nonzero(array)
        model["Priors"][i] = count / array.size
        for j in range(d):
            intersect = np.dot(array,train_data[j,:])
            num = np.count_nonzero(intersect)
            model["Probs"][j,i] = (num+1)/(count+2)
    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    n = data.shape[1]
    key = []
    priors = model["Priors"]
    probs = model["Probs"].T.dot(data)
    for i in range(n):
        x = probs[:,i]
        for j,p in enumerate(priors):
            probs[j,i] *= p
        probabilites = np.argmax(probs[:,i])
        key.append(probabilites)
    return key
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

    d, n = train_data.shape
    num_classes = labels.size

    model = {
        "Labels": labels,
        "Priors": np.empty(shape=(num_classes,1)),
        "Probs": np.empty(shape=(d,num_classes))
    }
    i = 0
    for l in labels:
        array = train_labels == l
        count = np.count_nonzero(array)
        model["Priors"][i,0] = count / array.size
        for j in range(d):
            intersect = np.dot(array,train_data[j,:])
            num = np.count_nonzero(intersect)
            model["Probs"][j,i] = (num+1)/(count+2)

        i += 1

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
    d,n = data.shape
    key = []
    labels = model["Labels"]
    probs = model["Probs"]
    priors = model["Priors"]
    for i in range(n):
        x = data[:,i]
        probabilites = np.zeros(labels.size)
        for index,c in enumerate(labels):
            for ind,att in enumerate(x):
                prob = probs[ind,index]
                if att:
                    probabilites[index] += np.log(prob)
                else:
                    prob = 1 - prob
                    probabilites[index] += np.log(prob)
            probabilites += np.log(priors[index])
        key.append(labels[np.argmax(probabilites)])

    return key
import numpy as np
from keras.metrics import top_k_categorical_accuracy
from scipy.signal import gaussian, find_peaks
from scipy.ndimage import filters
import scipy


def top_k_categorical_accuracy_asn(y_pred, y_true, k = 5, num_targets=1):
    """
    Imitates keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5) for numpy arrays
    :param y_pred: prediction by the network
    :param y_true: correct prediction
    :param k: top-k
    :return: accuracy
    Last updated: 20.11.18
    """
    if num_targets == 1:
        target = np.argmax(y_true, axis=-1)
        target = target[:,np.newaxis]
    elif num_targets == 2:
        target = np.argsort(-y_true, axis=-1)[:,:2]
        if np.any(y_true==2):
            idx = np.where(y_true==2)
            target[idx[0], 1] = target[idx[0], 0] # as a missing value
    elif num_targets == 3:
        target = np.argsort(-y_true, axis=-1)[:, :3]
        if np.any(y_true == 2):
            idx = np.where(y_true == 2)
            target[idx[0], 2] = target[idx[0], 0]
        if np.any(y_true==3):
            idx = np.where(y_true == 3)
            target[idx[0], 1:3] = target[idx[0], 0]

    y_pred_ordered = np.argsort(-y_pred, axis=-1)
    out = np.zeros((len(y_pred), num_targets), dtype=bool)
    for i in range(len(y_pred)):
        for j in range(num_targets):
            out[i,j]=np.any(y_pred_ordered[i, :k] == target[i,j])

    return np.mean(out)


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def tp_rate(y_predicted, y_true):
    return np.sum(y_predicted[y_true == 1]) / np.sum(y_true == 1)


def fp_rate(y_predicted, y_true):
    return np.sum(y_predicted[y_true == 0]) / np.sum(y_true == 0)



# Signal detection measures for binary tasks
def compute_SDTmeasures(predictions, labels, adjusted = False):
    """
    adapted from https://neurokit.readthedocs.io/en/latest/_modules/neurokit/statistics/routines.html

    :param predictions: 0 or 1 per trials, thus N,1
    :param labels: Same as for predictions
    :return: d', c & beta
    """

    # translate into hits
    n_Hit = np.sum((predictions> 0.5) & (labels ==1))
    n_FA = np.sum((predictions> 0.5) & (labels!=1))

    n_Miss = np.sum((predictions<0.5) & (labels==1))
    n_CR = np.sum((predictions<0.5) & (labels!=1))


    if adjusted == True:
        # Adjusted ratios
        hit_rate = (n_Hit + 0.5) / ((n_Hit + 0.5) + n_Miss + 1)
        fa_rate = (n_FA + 0.5) / ((n_FA + 0.5) + n_CR + 1)
    else:
        # Ratios
        hit_rate = n_Hit / (n_Hit + n_Miss)
        fa_rate = n_FA / (n_FA + n_CR)

    # dprime
    dprime = scipy.stats.norm.ppf(hit_rate) - scipy.stats.norm.ppf(fa_rate)

    # beta
    zhr = scipy.stats.norm.ppf(hit_rate)
    zfar = scipy.stats.norm.ppf(fa_rate)
    beta = np.exp(-zhr * zhr / 2 + zfar * zfar / 2)

    # aprime
    #a = 1 / 2 + ((hit_rate - fa_rate) * (1 + hit_rate - fa_rate) / (4 * hit_rate * (1 - fa_rate)))
    #b = 1 / 2 - ((fa_rate - hit_rate) * (1 + fa_rate - hit_rate) / (4 * fa_rate * (1 - hit_rate)))

    #if fa_rate > hit_rate:
    #    aprime = b
    #elif fa_rate < hit_rate:
    #    aprime = a
    #else:
    #    aprime = 0.5

    # bppd
    #bppd = ((1 - hit_rate) * (1 - fa_rate) - hit_rate * fa_rate) / ((1 - hit_rate) * (1 - fa_rate) + hit_rate * fa_rate)

    # c
    c = -(scipy.stats.norm.ppf(hit_rate) + scipy.stats.norm.ppf(fa_rate)) / 2

    parameters = dict(dprime=dprime, beta=beta, c=c, hit_rate=hit_rate, fa_rate=fa_rate)
    return (parameters)


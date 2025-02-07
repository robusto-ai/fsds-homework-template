import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    n = len(y_true)
    loss = 0
    for yt, yp in zip(y_true, y_pred):
        loss += yt * math.log(yp) + (1 - yt) * math.log(1 - yp)
    return -loss / n

def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
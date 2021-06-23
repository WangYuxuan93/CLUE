import csv
import sys
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def prf(preds, labels, pad_id=0):
    n_corr, n_pred, n_gold = 0, 0, 0
    for p, l in zip(preds, labels):
        if p != pad_id:
            n_pred += 1
        if l != pad_id:
            n_gold += 1
        if p != pad_id and l != pad_id and p == l:
            n_corr += 1
    if n_pred == 0:
        n_pred = 1
    if n_gold == 0:
        n_gold = 1
    n_corr = float(n_corr)
    recall = n_corr / n_gold
    precision = n_corr / n_pred
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return {
        "n_correct": n_corr,
        "n_gold": n_gold,
        "n_pred": n_pred,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    results = prf(preds=preds, labels=labels, pad_id=0)
    results["acc"] = acc
    return results

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)
    #return {"acc": simple_accuracy(preds, labels)}


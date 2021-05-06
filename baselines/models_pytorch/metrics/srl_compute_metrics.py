import csv
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    n_corr, n_tot = 0, 0
    for pred, label in zip(preds, labels):
        for p, l in zip(pred, label):
            n_tot += 1
            if p == l:
                n_corr += 1
    acc = float(n_corr) / n_tot
    return {
        "n_correct": n_corr,
        "n_total": n_tot,
        "acc": acc
    }

def prf(preds, labels, pad_id=1):
    n_corr, n_pred, n_gold = 0, 0, 0
    for pred, label in zip(preds, labels):
        for p, l in zip(pred, label):
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

def compute_metrics(task_name, preds, labels, debug=True):
    task_type = task_name.split('-')[2]
    assert len(preds) == len(labels)
    # Remove ignored index (special tokens)
    true_preds = [
        [p for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    if debug:
        #print ("preds:\n", preds)
        #print ("labels:\n", labels)
        print ("true_preds:\n", true_preds)
        print ("true_labels:\n", true_labels)
    if task_type == "arg":
        results = prf(preds=true_preds, labels=true_labels)
    elif task_type == "sense":
        # label['O'] == 1, only compute real predicate sense
        results = simple_accuracy(true_preds, true_labels)
    else:
        raise ValueError("Task: {} not defined in metrics!".format(task_name))
    return results
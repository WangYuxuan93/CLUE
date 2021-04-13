import csv
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

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

def compute_metrics(task_name, preds, labels, debug=False):
    task_type = task_name.split('-')[2]

    if task_type == "arg":
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
        results = prf(preds=true_preds, labels=true_labels)
        return results
    else:
        raise ValueError("Task: {} not defined in metrics!".format(task_name))
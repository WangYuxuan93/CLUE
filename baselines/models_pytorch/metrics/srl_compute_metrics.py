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

def compute_metrics(task_name, preds, labels, debug=False):
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


def prf_3d(preds, golds, pad_id=1):
    n_corr, n_pred, n_gold = 0, 0, 0
    for pred, gold in zip(preds, golds):
        for pline, gline in zip(pred, gold):
            for p, g in zip(pline, gline):
                # invalid entry
                if g == -100: continue
                if p != pad_id:
                    n_pred += 1
                if g != pad_id:
                    n_gold += 1
                if p != pad_id and g != pad_id and p == g:
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


def compute_end2end_metrics(task_name, preds, gold_rels, debug=False):
    if debug:
        print ("preds:\n", preds)
        print ("gold:\n", gold_rels)
    prd_preds = [
        [p for (p, l) in zip(pred[:,0], gold[:,0]) if l != -100]
        for pred, gold in zip(preds, gold_rels)
    ]
    prd_labels = [
        [l for (p, l) in zip(pred[:,0], gold[:,0]) if l != -100]
        for pred, gold in zip(preds, gold_rels)
    ]
    if debug:
        print ("prd_preds:\n", prd_preds)
        print ("prd_labels:\n", prd_labels)

    prd_results = simple_accuracy(prd_preds, prd_labels)
    arg_results = prf_3d([p[:,1:] for p in preds], [g[:,1:] for g in gold_rels], pad_id=1)

    results = {
        "n_prd_correct": prd_results["n_correct"],
        "n_prd_total": prd_results["n_total"],
        "prd_acc": prd_results["acc"],
        "n_arg_correct": arg_results["n_correct"],
        "n_arg_gold": arg_results["n_gold"],
        "n_arg_pred": arg_results["n_pred"],
        "arg_precision": arg_results["precision"],
        "arg_recall": arg_results["recall"],
        "arg_f1": arg_results["f1"],
        "n_correct": arg_results["n_correct"]+prd_results["n_correct"],
        "n_gold": arg_results["n_gold"]+prd_results["n_total"],
        "n_pred": arg_results["n_pred"]+prd_results["n_total"],
    }
    if results["n_gold"] == 0:
        results["recall"] = 0
    else:
        results["recall"] = results["n_correct"] / float(results["n_gold"])
    if results["n_pred"] == 0:
        results["precision"] = 0
    else:
        results["precision"] = results["n_correct"] / float(results["n_pred"])
    if results["precision"] + results["recall"] == 0:
        results["f1"] = 0
    else:
        results["f1"] = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"])
    return results

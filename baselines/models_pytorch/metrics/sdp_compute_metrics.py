import csv
import sys
import logging
import numpy as np
from collections import OrderedDict

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

def prf(pred_heads, pred_rels, gold_heads, gold_rels):
    n_labeled_corr, n_unlabeled_corr, n_pred, n_gold = 0, 0, 0, 0
    assert len(pred_heads) == len(gold_heads)
    sid = 0
    for pred, gold in zip(pred_heads, gold_heads):
        assert len(pred) == len(gold)
        seq_len = len(gold)
        for i in range(1, seq_len): # current id
            for j in range(seq_len): # head id
                if gold[i][j] == 1:
                    n_gold += 1
                if pred[i][j] == 1:
                    n_pred += 1
                if gold[i][j] == 1 and pred[i][j] == 1:
                    n_unlabeled_corr += 1
                    if gold_rels[sid][i][j] == pred_rels[sid][i][j]:
                        n_labeled_corr += 1
        sid += 1

    n_labeled_corr = float(n_labeled_corr)
    n_unlabeled_corr = float(n_unlabeled_corr)
    n_gold = 1e-9 if n_gold == 0 else n_gold
    n_pred = 1e-9 if n_pred == 0 else n_pred

    ur = n_unlabeled_corr / n_gold
    up = n_unlabeled_corr / n_pred
    lr = n_labeled_corr / n_gold
    lp = n_labeled_corr / n_pred
    if up + ur == 0:
        uf = 0
    else:
        uf = 2 * (up * ur) / (up + ur)
    if lp + lr == 0:
        lf = 0
    else:
        lf = 2 * (lp * lr) / (lp + lr)
    return OrderedDict({
        "lf": lf,
        "uf": uf,
        "lp": lp,
        "lr": lr,
        "up": up,
        "ur": ur,
        "n_labeled_corr": n_labeled_corr,
        "n_unlabeled_corr": n_unlabeled_corr,
        "n_gold": n_gold,
        "n_pred": n_pred
    })

def compute_metrics(pred_heads, pred_rels, gold_heads, gold_rels, debug=False):
    if debug:
        print ("pred_heads:\n", pred_heads)
        print ("gold_heads:\n", gold_heads)
        print ("pred_rels:\n", pred_rels)
        print ("gold_rels:\n", gold_rels)

    results = prf(pred_heads, pred_rels, gold_heads, gold_rels)

    return results
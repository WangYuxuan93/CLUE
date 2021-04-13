# -*- coding: utf-8 -*-

import logging
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from .utils import DataProcessor
from processors.processor import cached_features_filename
from .common import conll09_chinese_syntax_label_mapping, conll09_english_syntax_label_mapping

logger = logging.getLogger(__name__)

class SrlProcessor(DataProcessor):
    def __init__(self, task):
        self.task = task
        if task is None:
            self.lan = 'zh'
        else:
            self.lan = task.split('-')[1]

    def _read_conll(self, filename):
        sents = []
        with open(filename, 'r') as f:
            data = f.read().strip().split("\n\n")
            for sent in data:
                lines = sent.strip().split("\n")
                sents.append([line.split() for line in lines])
        return sents

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "test.txt")), "test", use_pos=True)

    def get_labels(self):
        """See base class."""
        raise NameError("get_labels() is not defined!")

    def get_syntax_label_map(self):
        if self.lan == 'zh':
            self.syntax_label_map = conll09_chinese_syntax_label_mapping
        elif self.lan == 'en':
            self.syntax_label_map = conll09_english_syntax_label_mapping
        # label for inner-word arc
        self.syntax_label_map['<WORD>'] = len(self.syntax_label_map)
        return self.syntax_label_map

    def get_pred_ids(self, sent):
        pred_ids = []
        pred_senses = []
        for i, line in enumerate(sent):
            if line[12] == 'Y':
                assert line[13] != '-'
                pred_ids.append(i)
                pred_senses.append(line[13])
        return pred_ids, pred_senses

    def _create_examples(self, sents, set_type, use_pos=False):
        """Creates examples for the training and dev sets."""
        raise NameError("_create_examples() is not defined!")

def get_word2token_map(word_ids, lengths, debug=False):
    assert len(word_ids) == len(lengths)
    wid2tid_list = []
    for wids, l in zip(word_ids, lengths):
        wid2tid = []
        prev_word = None
        for i, w in enumerate(wids):
            if i >= l: break
            if len(wid2tid) == 0 or w != prev_word:
                wid2tid.append([i])
                prev_word = w
            else:
                wid2tid[-1].append(i)
        wid2tid_list.append(wid2tid)
        if debug:
            print ("wids:\n", wids)
            print ("wid2tid:\n", wid2tid)
    return wid2tid_list

def align_flatten_heads(
        attention_mask,
        word_ids,
        flatten_heads,
        flatten_rels,
        max_length=128,
        syntax_label_map=None,
        expand_type="word",
        debug=False
    ):
        #print ("attention_mask:\n", attention_mask)
        lengths = [sum(mask) for mask in attention_mask]
        #print ("lengths:\n", lengths)
        #print ("word_ids:\n", word_ids)
        wid2tid_list = get_word2token_map(word_ids, lengths)

        heads_list = []
        rels_list = []
        for i in range(len(flatten_heads)):
            if debug:
                print ("word_ids:\n", word_ids[i])
                print ("wid2tid:\n", wid2tid_list[i])
                print ("flatten_heads:\n", flatten_heads[i])
                print ("flatten_rels:\n", flatten_rels[i])
            
            heads = torch.zeros(max_length, max_length, dtype=torch.long)
            rels = torch.zeros(max_length, max_length, dtype=torch.long)
            wid2tid = wid2tid_list[i]
            if "copy" in expand_type:
                head_ids = flatten_heads[i]
                # copy the arc from first char of the head to all chars consisting its children
                for child_id, head_id in enumerate(head_ids):
                    label = flatten_rels[i][child_id]
                    # add the first [CLS] token
                    child_id = child_id + 1
                    # head_id do not need since it is already 1 greater
                    head_id = head_id
                    head_ids = wid2tid[head_id]
                    # get the first token of the head word
                    token_head_id = head_ids[0]
                    child_ids = wid2tid[child_id]
                    for child_id in child_ids:
                        # ignore out of range arcs
                        if child_id < max_length and token_head_id < max_length:
                            heads[child_id][token_head_id] = 1
                            rels[child_id][token_head_id] = syntax_label_map[label]
            if debug:
                torch.set_printoptions(profile="full")
                print ("heads:\n", heads)
                print ("rels:\n", rels)

            if "word" in expand_type:
                # add arc with word_label from following chars to the first char of each word
                for tids in wid2tid:
                    if len(tids) > 1:
                        start_id = tids[0]
                        for cid in tids[1:]:
                            heads[cid][start_id] = 1
                            rels[cid][start_id] = syntax_label_map["<WORD>"]
                if debug:
                    print ("heads (word arc):\n", heads)
                    print ("rels (word arc):\n", rels)
                    #exit()

            heads_list.append(heads.to_sparse())
            rels_list.append(rels.to_sparse())
            # delete dense tensor to save mem
            del heads
            del rels

        heads = torch.stack(heads_list, dim=0)
        rels = torch.stack(rels_list, dim=0)

        if debug:
            print ("heads:\n", heads)
            print ("rels:\n", rels)
            exit()

        return heads, rels
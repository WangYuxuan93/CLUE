# -*- coding: utf-8 -*-

import logging
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from .utils import DataProcessor
from .common import conll09_chinese_label_mapping, conll09_english_label_mapping
from processors.processor import cached_features_filename
from processors.srl_processor import SrlProcessor, align_flatten_heads

logger = logging.getLogger(__name__)

class InputArgumentLabelExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, sid, tokens_a, pred_id, 
                 tokens_b=None, labels=None, pos_tags=None, 
                 syntax_heads=None, syntax_rels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            label: (Optional) list. The srl label for each of the argument
        """
        self.guid = guid
        self.sid = sid
        self.pred_id = pred_id # predicate idx
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.labels = labels
        self.pos_tags = pos_tags
        self.syntax_heads = syntax_heads
        self.syntax_rels = syntax_rels

    def show(self):
        logger.info("guid={}, sid={}, pred_id={}".format(self.guid, self.sid, self.pred_id))
        logger.info("tokens_a={}, tokens_b={}".format(self.tokens_a, self.tokens_b))
        logger.info("labels={}".format(self.labels))
        logger.info("pos_tags={}".format(self.pos_tags))


class InputArgumentLabelFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, predicate_mask, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.predicate_mask = predicate_mask
        self.label_ids = label_ids
        

class InputParsedArgumentLabelFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, predicate_mask, label_ids, heads, rels, dists=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.predicate_mask = predicate_mask
        self.label_ids = label_ids
        self.heads = heads
        self.rels = rels
        self.dists = dists


def argument_label_collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    num_items = len(batch[0])
    all_heads, all_rels, all_dists = None, None, None
    if num_items == 5:
        all_input_ids, all_attention_mask, all_token_type_ids, all_predicate_mask, all_labels = map(torch.stack, zip(*batch))
    elif num_items == 7:
        all_input_ids, all_attention_mask, all_token_type_ids, all_predicate_mask, all_labels, all_heads, all_rels = map(torch.stack, zip(*batch))
    elif num_items == 8:
        all_input_ids, all_attention_mask, all_token_type_ids, all_predicate_mask, all_labels, all_heads, all_rels, all_dists = map(torch.stack, zip(*batch))
    max_len = max(all_attention_mask.sum(-1)).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_predicate_mask = all_predicate_mask[:, :max_len]
    all_labels = all_labels[:, :max_len]
    if num_items >= 7:
        if all_heads.is_sparse:
            all_heads = all_heads.to_dense()
            all_rels = all_rels.to_dense()
        if len(all_heads.size()) == 3:
            all_heads = all_heads[:, :max_len, :max_len]
        elif len(all_heads.size()) == 4:
            all_heads = all_heads[:, :, :max_len, :max_len]
        all_rels = all_rels[:, :max_len, :max_len]
    if num_items == 8:
        if all_dists.is_sparse:
            all_dists = all_dists.to_dense()
        all_dists = all_dists[:, :max_len, :max_len]
    
    batch = {}
    batch["input_ids"] = all_input_ids
    batch["attention_mask"] = all_attention_mask
    batch["token_type_ids"] = all_token_type_ids
    batch["predicate_mask"] = all_predicate_mask
    batch["labels"] = all_labels
    batch["heads"] = all_heads
    batch["rels"] = all_rels
    batch["dists"] = all_dists
    return batch

    #return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_heads, all_rels, all_dists


class ArgumentLabelProcessor(SrlProcessor):
    def __init__(self, task):
        self.task = task
        if task is None:
            self.lan = 'zh'
        else:
            self.lan = task.split('-')[1]

    def get_labels(self):
        """See base class."""
        if self.lan == 'zh':
            self.label_map = conll09_chinese_label_mapping
        elif self.lan == 'en':
            self.label_map = conll09_english_label_mapping
        return list(self.label_map.keys())

    def _create_examples(self, sents, set_type, use_pos=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, sent) in enumerate(sents):
            sid = "%s-%s" % (set_type, i)
            pred_ids, _ = self.get_pred_ids(sent)
            tokens_a = [line[1] for line in sent]
            if use_pos:
                pos_tags = [line[5] for line in sent]
            else:
                pos_tags = None
            heads = [int(line[8]) for line in sent]
            rels = [line[10] for line in sent]
            for j, pred_id in enumerate(pred_ids): # the j-th predicate
                guid = "%s-%s" % (set_type, len(examples))
                tokens_b = [tokens_a[pred_id]]
                labels = [line[14+j] if line[14+j] != '_' else 'O' for line in sent]
                examples.append(
                    InputArgumentLabelExample(guid=guid, sid=sid, tokens_a=tokens_a, pred_id=pred_id, 
                                    tokens_b=tokens_b, labels=labels, pos_tags=pos_tags,
                                    syntax_heads=heads, syntax_rels=rels))
        return examples


def convert_examples_to_features(
        examples, 
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True
    ):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: CLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    processor = ArgumentLabelProcessor(task)
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    tokenized_inputs = tokenizer(
            [example.tokens_a for example in examples],
            [example.tokens_b for example in examples],
            padding='max_length',
            max_length=max_length,
            is_split_into_words=True)

    features = []
    #labels = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        word_ids = tokenized_inputs.word_ids(batch_index=ex_index)
        token_type_ids = tokenized_inputs['token_type_ids'][ex_index]
        attention_mask = tokenized_inputs['attention_mask'][ex_index]
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            # argument must be in tokens_a and not pad
            elif not (attention_mask[len(label_ids)] == 1 and token_type_ids[len(label_ids)]==0):
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[example.labels[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        #labels.append(label_ids)
        predicate_mask = np.zeros(max_length, dtype=np.int32)
        token_pred_ids = tokenized_inputs.word_to_tokens(ex_index, example.pred_id)
        # use the first token as predicate
        predicate_mask[token_pred_ids[0]] = 1

        if ex_index < 5:
            logger.info("*** Example ***")
            #logger.info("guid: %s" % (example.guid))
            #logger.info("sid: %s" % (example.sid))
            example.show()
            logger.info("word_ids: %s" % (" ".join([str(x) for x in word_ids])))
            logger.info("input_ids: %s" % " ".join([str(x) for x in tokenized_inputs['input_ids'][ex_index]]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in tokenized_inputs['attention_mask'][ex_index]]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in tokenized_inputs['token_type_ids'][ex_index]]))
            logger.info("pred_id: %d, token_pred_id: %d, predicate_mask: %s" % (example.pred_id, token_pred_ids[0], " ".join([str(x) for x in predicate_mask])))
            logger.info("labels: %s (ids = %s)" % (" ".join(example.labels), " ".join([str(l) for l in label_ids])))

        features.append(
            InputArgumentLabelFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                          attention_mask=tokenized_inputs['attention_mask'][ex_index],
                          token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                          predicate_mask=predicate_mask,
                          label_ids=label_ids))
    return features


def convert_parsed_examples_to_features(
        examples, 
        tokenizer,
        parser,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        use_gold_syntax=True,
        expand_type="word",
        align_type="nltk",
        return_tensor=True,
        compute_dist=False,
        return_graph_mask=False, 
        n_mask=3, 
        mask_types=["parent","child"]
    ):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: CLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    processor = ArgumentLabelProcessor(task)
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    tokenized_inputs = tokenizer(
            [example.tokens_a for example in examples],
            [example.tokens_b for example in examples],
            padding='max_length',
            max_length=max_length,
            is_split_into_words=True)
    
    if use_gold_syntax:
        heads, rels = align_flatten_heads(
                        attention_mask=tokenized_inputs['attention_mask'],
                        word_ids=[tokenized_inputs.word_ids(i) for i in range(len(examples))],
                        flatten_heads=[example.syntax_heads for example in examples],
                        flatten_rels=[example.syntax_rels for example in examples],
                        max_length=max_length,
                        syntax_label_map=processor.get_syntax_label_map(),
                        expand_type=expand_type,
                    )
    else:
        heads, rels = parser.parse_bpes(
                        tokenized_inputs['input_ids'],
                        tokenized_inputs['attention_mask'],
                        has_b=examples[0].text_b is not None,
                        expand_type=expand_type,
                        max_length=max_length, 
                        align_type=align_type, 
                        return_tensor=return_tensor, 
                        sep_token_id=tokenizer.sep_token_id,
                        return_graph_mask=return_graph_mask, 
                        n_mask=n_mask, 
                        mask_types=mask_types
                    )
    dists = None
    if compute_dist:
        dists = compute_distance(heads, attention_mask_list)

    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        word_ids = tokenized_inputs.word_ids(batch_index=ex_index)
        token_type_ids = tokenized_inputs['token_type_ids'][ex_index]
        attention_mask = tokenized_inputs['attention_mask'][ex_index]
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            # argument must be in tokens_a and not pad
            elif not (attention_mask[len(label_ids)] == 1 and token_type_ids[len(label_ids)]==0):
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[example.labels[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        #labels.append(label_ids)
        predicate_mask = np.zeros(max_length, dtype=np.int32)
        token_pred_ids = tokenized_inputs.word_to_tokens(ex_index, example.pred_id)
        # use the first token as predicate
        predicate_mask[token_pred_ids[0]] = 1

        if ex_index < 5:
            logger.info("*** Example ***")
            #logger.info("guid: %s" % (example.guid))
            #logger.info("sid: %s" % (example.sid))
            example.show()
            logger.info("word_ids: %s" % (" ".join([str(x) for x in word_ids])))
            logger.info("input_ids: %s" % " ".join([str(x) for x in tokenized_inputs['input_ids'][ex_index]]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in tokenized_inputs['attention_mask'][ex_index]]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in tokenized_inputs['token_type_ids'][ex_index]]))
            logger.info("pred_id: %d, token_pred_id: %d, predicate_mask: %s" % (example.pred_id, token_pred_ids[0], " ".join([str(x) for x in predicate_mask])))
            logger.info("labels: %s (ids = %s)" % (" ".join(example.labels), " ".join([str(l) for l in label_ids])))
            torch.set_printoptions(profile="full")
            print ("\nheads:\n", heads[ex_index])
            print ("\nrels:\n", rels[ex_index])
            if dists:
                print ("\ndists:\n", dists[ex_index])

        if compute_dist:
            features.append(
                InputParsedArgumentLabelFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                              attention_mask=tokenized_inputs['attention_mask'][ex_index],
                              token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                              predicate_mask=predicate_mask,
                              label_ids=label_ids,
                              heads=heads[ex_index] if heads.is_sparse else heads[ex_index].to_sparse(),
                              rels=rels[ex_index] if rels.is_sparse else rels[ex_index].to_sparse(),
                              dists=dists[ex_index] if dists.is_sparse else dists[ex_index].to_sparse()))
        else:
            features.append(
                InputParsedArgumentLabelFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                              attention_mask=tokenized_inputs['attention_mask'][ex_index],
                              token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                              predicate_mask=predicate_mask,
                              label_ids=label_ids,
                              heads=heads[ex_index] if heads.is_sparse else heads[ex_index].to_sparse(),
                              rels=rels[ex_index] if rels.is_sparse else rels[ex_index].to_sparse()))
    return features
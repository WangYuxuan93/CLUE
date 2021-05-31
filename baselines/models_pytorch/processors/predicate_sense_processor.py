# -*- coding: utf-8 -*-

import logging
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from .utils import DataProcessor
from .mappings.conll09_srl_pipeline_mapping import conll09_chinese_sense_mapping, conll09_english_sense_mapping
from .mappings.upb_srl_pipeline_mapping import upb_chinese_sense_mapping
from processors.processor import cached_features_filename
from processors.srl_processor import SrlProcessor, align_flatten_heads
from processors.srl_processor import prepare_word_level_input, flatten_heads_to_matrix

logger = logging.getLogger(__name__)

class InputPredicateSenseFeatures(object):
    """A single set of features of data."""

    def __init__(
            self, 
            input_ids, 
            attention_mask, 
            token_type_ids, 
            predicate_mask, 
            label_ids,
            word_mask=None,
            first_ids=None
        ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.predicate_mask = predicate_mask
        self.label_ids = label_ids
        self.word_mask = word_mask
        self.first_ids = first_ids


class InputParsedPredicateSenseFeatures(object):
    """A single set of features of data."""

    def __init__(
            self, 
            input_ids, 
            attention_mask, 
            token_type_ids, 
            predicate_mask, 
            label_ids, 
            heads, 
            rels,
            word_mask=None,
            first_ids=None,
            dists=None
        ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.predicate_mask = predicate_mask
        self.label_ids = label_ids
        self.heads = heads
        self.rels = rels
        self.word_mask = word_mask
        self.first_ids = first_ids
        self.dists = dists


class PredicateSenseProcessor(SrlProcessor):
    def __init__(self, task):
        super().__init__(task)

    def get_labels(self):
        """See base class."""
        if self.task.startswith('conll09-zh'):
            self.label_map = conll09_chinese_sense_mapping
        elif self.task.startswith('conll09-en'):
            self.label_map = conll09_english_sense_mapping
        elif self.task.startswith('upb-zh'):
            self.label_map = upb_chinese_sense_mapping
        return list(self.label_map.keys())

def convert_examples_to_features(
        examples, 
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        is_word_level=False,
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
    processor = PredicateSenseProcessor(task)
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    tokenized_inputs = tokenizer(
            [example.words for example in examples],
            padding='max_length',
            max_length=max_length,
            is_split_into_words=True,
            return_token_type_ids=True)
    print ("tokenized_inputs:\n", tokenized_inputs)
    features = []
    word_masks, first_ids_list, word_lens = prepare_word_level_input(
        attention_mask=[mask for mask in tokenized_inputs['attention_mask']],
        word_ids=[tokenized_inputs.word_ids(i) for i in range(len(examples))],
        tokens=[example.words for example in examples])
    max_word_len = max(word_lens)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        word_ids = tokenized_inputs.word_ids(batch_index=ex_index)
        # default is empty -100
        label_ids = np.ones(max_word_len, dtype=np.int32) * -100
        for i in range(len(example.pred_senses)):
            label_id = label_map[example.pred_senses[i]]
            if label_id == 0:
                label_id = -100
            label_ids[i] = label_id
        predicate_mask = np.zeros(max_word_len, dtype=np.int32)
        for word_pred_id in example.pred_ids:
            predicate_mask[word_pred_id] = 1

        if ex_index < 2:
            logger.info("*** Example ***")
            #logger.info("guid: %s" % (example.guid))
            #logger.info("sid: %s" % (example.sid))
            example.show()
            logger.info("word_ids: %s" % (" ".join([str(x) for x in word_ids])))
            logger.info("input_ids: %s" % " ".join([str(x) for x in tokenized_inputs['input_ids'][ex_index]]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in tokenized_inputs['attention_mask'][ex_index]]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in tokenized_inputs['token_type_ids'][ex_index]]))
            logger.info("pred_ids: %s, predicate_mask: %s" % (" ".join([str(x) for x in example.pred_ids]), " ".join([str(x) for x in predicate_mask])))
            logger.info("labels: %s (ids = %s)" % (" ".join(example.pred_senses), " ".join([str(l) for l in label_ids])))
            logger.info("word_mask: {}".format(word_masks[ex_index]))
            logger.info("first_ids: {}".format(first_ids_list[ex_index]))

        features.append(
            InputPredicateSenseFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                          attention_mask=tokenized_inputs['attention_mask'][ex_index],
                          token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                          predicate_mask=predicate_mask,
                          label_ids=label_ids,
                          word_mask=word_masks[ex_index],
                          first_ids=first_ids_list[ex_index]))

    return features


def convert_parsed_examples_to_features(
        examples, 
        tokenizer,
        parser,
        max_length=512,
        task=None,
        label_list=None,
        is_word_level=False,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        official_syntax_type="gold",
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
    processor = PredicateSenseProcessor(task)
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    tokenized_inputs = tokenizer(
            [example.words for example in examples],
            padding='max_length',
            max_length=max_length,
            is_split_into_words=True,
            return_token_type_ids=True)
    
    word_masks, first_ids_list, word_lens = prepare_word_level_input(
            attention_mask=[mask for mask in tokenized_inputs['attention_mask']],
            word_ids=[tokenized_inputs.word_ids(i) for i in range(len(examples))],
            tokens=[example.words for example in examples])

    features = []
    if is_word_level:
        if official_syntax_type == "gold":
            heads, rels = flatten_heads_to_matrix(
                            word_masks=word_masks,
                            flatten_heads=[example.gold_heads for example in examples],
                            flatten_rels=[example.gold_rels for example in examples],
                            syntax_label_map=processor.get_syntax_label_map(),
                        )
        elif official_syntax_type == "pred":
            heads, rels = flatten_heads_to_matrix(
                            word_masks=word_masks,
                            flatten_heads=[example.pred_heads for example in examples],
                            flatten_rels=[example.pred_rels for example in examples],
                            syntax_label_map=processor.get_syntax_label_map(),
                        )
        else:
            print ("official_syntax_type: {} not defined.".format(official_syntax_type))
    else:
        if official_syntax_type == "gold":
            heads, rels = align_flatten_heads(
                            attention_mask=tokenized_inputs['attention_mask'],
                            word_ids=[tokenized_inputs.word_ids(i) for i in range(len(examples))],
                            flatten_heads=[example.gold_heads for example in examples],
                            flatten_rels=[example.gold_rels for example in examples],
                            max_length=max_length,
                            syntax_label_map=processor.get_syntax_label_map(),
                            expand_type=expand_type,
                            words_list=[example.words for example in examples]
                        )
        elif official_syntax_type == "pred":
            heads, rels = align_flatten_heads(
                            attention_mask=tokenized_inputs['attention_mask'],
                            word_ids=[tokenized_inputs.word_ids(i) for i in range(len(examples))],
                            flatten_heads=[example.pred_heads for example in examples],
                            flatten_rels=[example.pred_rels for example in examples],
                            max_length=max_length,
                            syntax_label_map=processor.get_syntax_label_map(),
                            expand_type=expand_type,
                            words_list=[example.words for example in examples]
                        )
        else:
            heads, rels = parser.parse_bpes(
                            tokenized_inputs['input_ids'],
                            tokenized_inputs['attention_mask'],
                            has_b=False,
                            expand_type=expand_type,
                            max_length=max_length, 
                            align_type=align_type, 
                            return_tensor=return_tensor, 
                            sep_token_id=tokenizer.sep_token_id,
                            return_graph_mask=return_graph_mask, 
                            n_mask=n_mask, 
                            mask_types=mask_types
                        )

    max_word_len = max(word_lens)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        word_ids = tokenized_inputs.word_ids(batch_index=ex_index)
        label_ids = np.ones(max_word_len, dtype=np.int32) * -100
        for i in range(len(example.pred_senses)):
            label_id = label_map[example.pred_senses[i]]
            if label_id == 0:
                label_id = -100
            label_ids[i] = label_id
        predicate_mask = np.zeros(max_word_len, dtype=np.int32)
        for word_pred_id in example.pred_ids:
            predicate_mask[word_pred_id] = 1

        if ex_index < 2:
            logger.info("*** Example ***")
            #logger.info("guid: %s" % (example.guid))
            #logger.info("sid: %s" % (example.sid))
            example.show()
            logger.info("word_ids: %s" % (" ".join([str(x) for x in word_ids])))
            logger.info("input_ids: %s" % " ".join([str(x) for x in tokenized_inputs['input_ids'][ex_index]]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in tokenized_inputs['attention_mask'][ex_index]]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in tokenized_inputs['token_type_ids'][ex_index]]))
            logger.info("pred_ids: %s, predicate_mask: %s" % (" ".join([str(x) for x in example.pred_ids]), " ".join([str(x) for x in predicate_mask])))
            logger.info("labels: %s (ids = %s)" % (" ".join(example.pred_senses), " ".join([str(l) for l in label_ids])))
            logger.info("word_mask: {}".format(word_masks[ex_index]))
            logger.info("first_ids: {}".format(first_ids_list[ex_index]))
            torch.set_printoptions(profile="full")
            print ("\nheads:\n", heads[ex_index])
            print ("\nrels:\n", rels[ex_index])

        features.append(
            InputParsedPredicateSenseFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                          attention_mask=tokenized_inputs['attention_mask'][ex_index],
                          token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                          predicate_mask=predicate_mask,
                          label_ids=label_ids,
                          heads=heads[ex_index] if heads.is_sparse else heads[ex_index].to_sparse(),
                          rels=rels[ex_index] if rels.is_sparse else rels[ex_index].to_sparse(),
                          word_mask=word_masks[ex_index],
                          first_ids=first_ids_list[ex_index]))
    return features
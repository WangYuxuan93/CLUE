# -*- coding: utf-8 -*-

import logging
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from .utils import DataProcessor
from .mappings.conll09_srl_end2end_mapping import conll09_english_end2end_mapping, conll09_english_num_arg_label
from .mappings.conll09_srl_end2end_mapping import conll09_chinese_end2end_mapping, conll09_chinese_num_arg_label
from processors.processor import cached_features_filename
from processors.srl_processor import InputConll09Example, SrlProcessor, align_flatten_heads
from processors.srl_processor import prepare_word_level_input, flatten_heads_to_matrix

from neuronlp2.parser import Parser
from neuronlp2.sdp_parser import SDPParser


logger = logging.getLogger(__name__)

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    num_items = len(batch[0])
    all_heads, all_rels, all_dists = None, None, None
    if num_items == 8:
        all_input_ids, all_attention_mask, all_token_type_ids, all_predicate_mask, all_srl_heads, all_srl_rels, \
        all_first_ids, all_word_mask = map(torch.stack, zip(*batch))
    elif num_items == 10:
        all_input_ids, all_attention_mask, all_token_type_ids, all_predicate_mask, all_srl_heads, all_srl_rels, \
        all_first_ids, all_word_mask, all_heads, all_rels = map(torch.stack, zip(*batch))
    max_len = max(all_attention_mask.sum(-1)).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]

    max_word_len = max(all_word_mask.sum(-1)).item()
    all_first_ids = all_first_ids[:, :max_word_len]
    all_word_mask = all_word_mask[:, :max_word_len]
    all_predicate_mask = all_predicate_mask[:, :max_word_len]
    all_srl_heads = all_srl_heads[:, :max_word_len, :max_word_len]
    all_srl_rels = all_srl_rels[:, :max_word_len, :max_word_len]

    if all_heads is not None:
        if list(all_heads.size())[-1] == list(all_input_ids.size())[-1]:
            # subword-level syntax matrix
            head_max_len = max_len
        else:
            # word-level syntax matrix
            head_max_len = max_word_len
        
        if all_heads.is_sparse:
            all_heads = all_heads.to_dense()
            all_rels = all_rels.to_dense()
        if len(all_heads.size()) == 3:
            all_heads = all_heads[:, :head_max_len, :head_max_len]
        elif len(all_heads.size()) == 4:
            all_heads = all_heads[:, :, :head_max_len, :head_max_len]
        all_rels = all_rels[:, :head_max_len, :head_max_len]
    if all_dists is not None:
        if all_dists.is_sparse:
            all_dists = all_dists.to_dense()
        all_dists = all_dists[:, :head_max_len, :head_max_len]
    
    batch = {}
    batch["input_ids"] = all_input_ids
    batch["attention_mask"] = all_attention_mask
    batch["token_type_ids"] = all_token_type_ids
    batch["predicate_mask"] = all_predicate_mask
    batch["srl_heads"] = all_srl_heads
    batch["srl_rels"] = all_srl_rels
    batch["first_ids"] = all_first_ids
    batch["word_mask"] = all_word_mask
    batch["heads"] = all_heads
    batch["rels"] = all_rels
    batch["dists"] = all_dists
    return batch


class InputSrlEnd2EndFeatures(object):
    """A single set of features of data."""

    def __init__(
            self, 
            input_ids, 
            attention_mask, 
            token_type_ids, 
            predicate_mask, 
            srl_heads,
            srl_rels,
            word_mask=None,
            first_ids=None
        ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.predicate_mask = predicate_mask
        self.srl_heads = srl_heads
        self.srl_rels = srl_rels
        self.word_mask = word_mask
        self.first_ids = first_ids


class InputParsedSrlEnd2EndFeatures(object):
    """A single set of features of data."""

    def __init__(
            self, 
            input_ids, 
            attention_mask, 
            token_type_ids, 
            predicate_mask,
            srl_heads, 
            srl_rels,
            source_heads,
            source_rels,
            word_mask=None,
            first_ids=None
        ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.predicate_mask = predicate_mask
        self.label_ids = label_ids
        self.srl_heads = srl_heads
        self.srl_rels = srl_rels
        self.heads = heads
        self.rels = rels
        self.word_mask = word_mask
        self.first_ids = first_ids


class SrlEnd2EndProcessor(SrlProcessor):
    def __init__(self, task):
        super().__init__(task)
        self.root_token = "<ROOT>"

    def get_labels(self):
        """See base class."""
        if self.lan == 'zh':
            self.label_map = conll09_chinese_end2end_mapping
        elif self.lan == 'en':
            self.label_map = conll09_english_end2end_mapping
        return list(self.label_map.keys())

    def get_arg_label_mask(self):
        mask = np.zeros(len(self.label_map))
        num_arg_label = conll09_chinese_num_arg_label if self.lan == 'zh' else conll09_english_num_arg_label
        for i in range(len(mask)):
            if i < num_arg_label:
                mask[i] = 1
        return mask

    def get_pred_label_mask(self):
        mask = 1 - self.get_arg_label_mask()
        return mask

    def get_pred_ids(self, sent):
        pred_ids = []
        pred_senses = []
        for i, line in enumerate(sent):
            if line[12] == 'Y':
                assert line[13] != '-'
                pred_ids.append(i)
                pred_senses.append(line[13])
        return pred_ids, pred_senses

    def _create_examples(self, sents, set_type, use_pos=False, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, sent) in enumerate(sents):
            sid = "%s-%s" % (set_type, i)
            pred_ids, _ = self.get_pred_ids(sent)
            words = [line[1] for line in sent]
            if use_pos:
                pos_tags = [line[5] for line in sent]
                pos_tags = [self.root_token] + pos_tags
            else:
                pos_tags = None
            heads = [int(line[8]) for line in sent]
            rels = [line[10] for line in sent]
            pred_heads = [int(line[9]) for line in sent]
            pred_rels = [line[11] for line in sent]
            pred_senses = [line[13].split('.')[1] if line[13] != '_' and line[12] == 'Y' else '<PAD>' for line in sent]
            arg_labels = []
            for j in range(len(pred_ids)):
                arg_labels.append([line[14+j] if line[14+j] != '_' else 'O' for line in sent])
            guid = "%s-%s" % (set_type, len(examples))

            # add <ROOT> at beginning
            pred_ids = [x+1 for x in pred_ids]
            words = [self.root_token] + words
            pred_senses = [self.root_token] + pred_senses
            arg_labels = [[self.root_token] + x for x in arg_labels]

            heads = [0] + [x+1 for x in heads]
            rels = [self.root_token] + rels
            pred_heads = [0] + [x+1 for x in pred_heads]
            pred_rels = [self.root_token] + pred_rels

            examples.append(
                InputConll09Example(guid=guid, sid=sid, words=words, pred_ids=pred_ids, 
                                pred_senses=pred_senses, arg_labels=arg_labels, pos_tags=pos_tags,
                                gold_heads=heads, gold_rels=rels, 
                                pred_heads=pred_heads, pred_rels=pred_rels))
        return examples

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
    processor = SrlEnd2EndProcessor(task)
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
        srl_heads = np.ones((max_word_len, max_word_len), dtype=np.int32) * -100
        srl_rels = np.ones((max_word_len, max_word_len), dtype=np.int32) * -100
        for pred_id, pred_label in enumerate(example.pred_senses):
            if pred_label not in ['<ROOT>','<PAD>','O']:
                srl_heads[0][pred_id] = 1
                pred_label = 'prd:'+pred_label
                srl_rels[0][pred_id] = label_map[pred_label]
        for i, pred_id in enumerate(example.pred_ids):
            arg_labels = example.arg_labels[i]
            for arg_id, arg_label in enumerate(arg_labels):
                # set label for <ROOT> -100, so it is automatically ignored
                if arg_label == '<ROOT>': continue
                srl_heads[pred_id][arg_id] = 0 if arg_label == 'O' else 1
                if arg_label not in ['<PAD>','O']:
                    arg_label = 'arg:'+arg_label
                srl_rels[pred_id][arg_id] = label_map[arg_label]
        srl_heads = srl_heads.transpose(1,0)
        srl_rels = srl_rels.transpose(1,0)

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
            logger.info("srl_heads:\n{}".format(srl_heads))
            logger.info("srl_rels:\n{}".format(srl_rels))
            logger.info("word_mask: {}".format(word_masks[ex_index]))
            logger.info("first_ids: {}".format(first_ids_list[ex_index]))

        features.append(
            InputSrlEnd2EndFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                          attention_mask=tokenized_inputs['attention_mask'][ex_index],
                          token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                          predicate_mask=predicate_mask,
                          srl_heads=srl_heads,
                          srl_rels=srl_rels,
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
    processor = SrlEnd2EndProcessor(task)
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
    
    features = []
    if is_word_level:
        word_masks, first_ids_list, word_lens = prepare_word_level_input(
            attention_mask=[mask for mask in tokenized_inputs['attention_mask']],
            word_ids=[tokenized_inputs.word_ids(i) for i in range(len(examples))],
            tokens=[example.words for example in examples])

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
                    label_id = label_map[example.pred_senses[word_idx]]
                    if label_id == 0:
                        label_id = -100
                    label_ids.append(label_id)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            #labels.append(label_ids)
            predicate_mask = np.zeros(max_length, dtype=np.int32)
            for word_pred_id in example.pred_ids:
                token_pred_ids = tokenized_inputs.word_to_tokens(ex_index, word_pred_id)
                # use the first token as predicate
                predicate_mask[token_pred_ids[0]] = 1

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
                torch.set_printoptions(profile="full")
                print ("\nheads:\n", heads[ex_index])
                print ("\nrels:\n", rels[ex_index])
                if dists:
                    print ("\ndists:\n", dists[ex_index])

            features.append(
                InputParsedPredicateSenseFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                              attention_mask=tokenized_inputs['attention_mask'][ex_index],
                              token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                              predicate_mask=predicate_mask,
                              label_ids=label_ids,
                              heads=heads[ex_index] if heads.is_sparse else heads[ex_index].to_sparse(),
                              rels=rels[ex_index] if rels.is_sparse else rels[ex_index].to_sparse()))
    return features



def load_and_cache_examples(args, task, tokenizer, data_type='train', return_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    task_type = task.split('-')[2]
    processor = SrlEnd2EndProcessor(task)
    #elif task_type == 'sense':

    output_mode = 'srl'
    cached_examples_file = os.path.join(args.data_dir, 'cached_examples_{}_{}'.format(data_type, str(task) if not args.is_word_level else str(task)+"-word-level"))
    if return_examples:
        if not os.path.exists(cached_examples_file):
            examples = processor.get_examples(args.data_dir, data_type)
            if args.local_rank in [-1, 0]:
                logger.info("Saving examples into cached file %s", cached_examples_file)
                torch.save(examples, cached_examples_file)
        else:
            logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
    # Load data features from cache or dataset file
    if args.official_syntax_type: # gold or pred official syntax
        parser_info = args.official_syntax_type+"-syntax"
        if args.is_word_level:
            parser_info += "-word-level"
        #parser_info = "gold-syntax"
        if args.parser_return_tensor:
            parser_info += "-3d"
        if args.parser_compute_dist:
            parser_info += "-dist"
        if args.parser_return_graph_mask:
            parser_info += "-mask-"+str(args.parser_n_mask)+"-"+"-".join(args.parser_mask_types)
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_parsed_{}_{}_{}'.format(
            data_type,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task),
            parser_info,
            args.parser_expand_type,
            args.parser_align_type))
        logger.info("Cached features filename: {}".format(cached_features_file))
    else:
        cached_features_file = cached_features_filename(args, task if not args.is_word_level else task+"-word-level", data_type=data_type)

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        examples = processor.get_examples(args.data_dir, data_type)
        """
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir, is_ood=is_ood)
        """

        if args.parser_model is not None:
            if args.parser_type == "dp":
                biaffine_parser = Parser(args.parser_model, pretrained_lm="roberta", lm_path=args.parser_lm_path,
                                    batch_size=args.parser_batch, parser_type=args.parser_type)
            elif args.parser_type == "sdp":
                biaffine_parser = SDPParser(args.parser_model, pretrained_lm="roberta", lm_path=args.parser_lm_path,
                                    batch_size=args.parser_batch, parser_type=args.parser_type)
        else:
            biaffine_parser = None

        if biaffine_parser is None and not args.official_syntax_type:
            features = convert_examples_to_features(examples,
                                            tokenizer,
                                            task=args.task_name,
                                            label_list=label_list,
                                            is_word_level=args.is_word_level,
                                            max_length=args.max_seq_length,
                                            output_mode=output_mode,
                                            pad_on_left=bool(args.model_type in ['xlnet']),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )
        else:
            features = convert_parsed_examples_to_features(examples,
                                                    tokenizer,
                                                    biaffine_parser,
                                                    task=args.task_name,
                                                    label_list=label_list,
                                                    is_word_level=args.is_word_level,
                                                    max_length=args.max_seq_length,
                                                    output_mode=output_mode,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                    official_syntax_type=args.official_syntax_type,
                                                    expand_type=args.parser_expand_type,
                                                    align_type=args.parser_align_type,
                                                    return_tensor=args.parser_return_tensor,
                                                    compute_dist=args.parser_compute_dist,
                                                    return_graph_mask=args.parser_return_graph_mask, 
                                                    n_mask=args.parser_n_mask, 
                                                    mask_types=args.parser_mask_types,
                                                    )
    
            del biaffine_parser

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_predicate_mask = torch.tensor([f.predicate_mask for f in features], dtype=torch.long)
    all_srl_heads = torch.tensor([f.srl_heads for f in features], dtype=torch.long)
    all_srl_rels = torch.tensor([f.srl_rels for f in features], dtype=torch.long)

    all_first_ids = torch.tensor([f.first_ids for f in features], dtype=torch.long)
    all_word_mask = torch.tensor([f.word_mask for f in features], dtype=torch.long)

    if args.parser_model is None and not args.official_syntax_type:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_predicate_mask, 
                                all_srl_heads, all_srl_rels, all_first_ids, all_word_mask)
    else:
        all_heads = torch.stack([f.heads for f in features])
        all_rels = torch.stack([f.rels for f in features])
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_predicate_mask,
                                all_srl_heads, all_srl_rels, all_first_ids, all_word_mask, all_heads, all_rels)

    if return_examples:
        return dataset, examples
    return dataset

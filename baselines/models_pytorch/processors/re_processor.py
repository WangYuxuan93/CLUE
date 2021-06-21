# -*- coding: utf-8 -*-

import logging
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from .utils import DataProcessor
from .mappings.sanwen_mapping import sanwen_label_mapping
from .mappings.finre_mapping import finre_label_mapping
from processors.processor import cached_features_filename
#from processors.srl_processor import InputConll09Example, SrlProcessor, align_flatten_heads
#from processors.srl_processor import prepare_word_level_input, flatten_heads_to_matrix

from processors.srl_processor import align_flatten_heads_sdp, flatten_heads_to_matrix_sdp
from .mappings.chinese_sdp_lv3_label_mapping import chinese_sdp_lv3_label_mapping

from neuronlp2.parser import Parser
from neuronlp2.sdp_parser import SDPParser


logger = logging.getLogger(__name__)

class InputReExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, 
                guid,
                text, 
                ent1,
                beg1,
                end1,
                ent2,
                beg2,
                end2,
                label=None):
        self.guid = guid
        self.text = text
        self.ent1 = ent1
        self.beg1 = beg1
        self.end1 = end1
        self.ent2 = ent2
        self.beg2 = beg2
        self.end2 = end2
        self.label = label

    def show(self):
        logger.info("guid={}".format(self.guid))
        logger.info("ent1={} ({}-{})".format(self.ent1, self.beg1, self.end1))
        logger.info("ent2={} ({}-{})".format(self.ent2, self.beg2, self.end2))
        logger.info("text={}".format(self.text))
        logger.info("label={}".format(self.label))


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
    # save all_input_ids to decise between word/subword-level 
    all_input_ids_ = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]

    max_word_len = max(all_word_mask.sum(-1)).item()
    all_first_ids = all_first_ids[:, :max_word_len]
    all_word_mask = all_word_mask[:, :max_word_len]
    all_predicate_mask = all_predicate_mask[:, :max_word_len]

    if all_srl_heads.is_sparse:
        all_srl_heads = all_srl_heads.to_dense()
    if all_srl_rels.is_sparse:
        all_srl_rels = all_srl_rels.to_dense()
    all_srl_heads = all_srl_heads[:, :max_word_len, :max_word_len]
    all_srl_rels = all_srl_rels[:, :max_word_len, :max_word_len]
    #print ("all_srl_rels:\n", all_srl_rels)
    neg_mask = torch.ones_like(all_srl_heads) * -100
    all_srl_heads = torch.where(all_srl_heads==0, neg_mask, all_srl_heads)
    all_srl_rels = torch.where(all_srl_rels==0, neg_mask, all_srl_rels)
    #print ("all_srl_rels:\n", all_srl_rels)
    if all_heads is not None:
        #print ("all_heads:", all_heads.size())
        #print ("all_input_ids:", all_input_ids.size())
        if list(all_heads.size())[-1] == list(all_input_ids.size())[-1]:
            # subword-level syntax matrix
            head_max_len = max_len
        else:
            # word-level syntax matrix
            head_max_len = max_word_len
        #print ("head_max_len=",head_max_len)
        
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
    batch["input_ids"] = all_input_ids_
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


class InputReFeatures(object):
    """A single set of features of data."""
    def __init__(
            self, 
            input_ids, 
            attention_mask, 
            token_type_ids,
            ent1_id, 
            ent2_id, 
            label=None
        ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.ent1_id = ent1_id
        self.ent2_id = ent2_id
        self.label = label


class InputParsedReFeatures(object):
    """A single set of features of data."""
    def __init__(
            self, 
            input_ids, 
            attention_mask, 
            token_type_ids, 
            heads,
            rels,
            ent1_id, 
            ent2_id, 
            label=None
        ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.heads = heads
        self.rels = rels
        self.ent1_id = ent1_id
        self.ent2_id = ent2_id
        self.label = label


class ReProcessor(DataProcessor):
    def __init__(self, task):
        #super().__init__(task)
        self.task = task
        self.root_token = "<ROOT>"

    def get_labels(self):
        """See base class."""
        if self.task.startswith('sanwen'):
            self.label_map = sanwen_label_mapping
        elif self.task.startswith('finre'):
            self.label_map = finre_label_mapping
        return list(self.label_map.keys())

    def _read_data(self, filename):
        sents = []
        with open(filename, 'r') as f:
            data = f.read().strip().split("\n")
            for line in data:
                items = line.strip().split("\t")
                try:
                    assert len(items) == 8
                except:
                    print ("Item mismatach: {}".format(line))
                sents.append(items)
        return sents

    def get_examples(self, data_dir, data_type="train"):
        """See base class."""
        filename = data_type+".txt"
        return self._create_examples(
            self._read_data(os.path.join(data_dir, filename)), data_type)

    def _create_examples(self, sents, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, sent) in enumerate(sents):
            guid = "%s-%s" % (set_type, i)
            ent1, beg1, end1, ent2, beg2, end2, label, text = sent

            examples.append(
                InputReExample(guid=guid, text=text, ent1=ent1, beg1=beg1, end1=end1, 
                                ent2=ent2, beg2=beg2, end2=end2, label=label))
            examples[-1].show()
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
    processor = ReProcessor(task)
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    tokenized_inputs = tokenizer(
            [[tokenizer.cls_token if x == '<ROOT>' else x for x in example.words] for example in examples],
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
        #srl_heads = np.ones((max_word_len, max_word_len), dtype=np.int32) * -100
        #srl_rels = np.ones((max_word_len, max_word_len), dtype=np.int32) * -100
        srl_heads = np.zeros((max_word_len, max_word_len), dtype=np.int32)
        srl_rels = np.zeros((max_word_len, max_word_len), dtype=np.int32)
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

        srl_heads = torch.from_numpy(srl_heads).long().to_sparse()
        srl_rels = torch.from_numpy(srl_rels).long().to_sparse()

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
            [[tokenizer.cls_token if x == '<ROOT>' else x for x in example.words] for example in examples],
            padding='max_length',
            max_length=max_length,
            is_split_into_words=True,
            return_token_type_ids=True)
    
    word_masks, first_ids_list, word_lens = prepare_word_level_input(
        attention_mask=[mask for mask in tokenized_inputs['attention_mask']],
        word_ids=[tokenized_inputs.word_ids(i) for i in range(len(examples))],
        tokens=[example.words for example in examples])
    max_word_len = max(word_lens)

    if is_word_level:
        # generate word-level syntax graphs, for top GNN
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
        # generate subword-level syntax graphs
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

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        word_ids = tokenized_inputs.word_ids(batch_index=ex_index)
        # default is empty -100
        #srl_heads = np.ones((max_word_len, max_word_len), dtype=np.int32) * -100
        #srl_rels = np.ones((max_word_len, max_word_len), dtype=np.int32) * -100
        srl_heads = np.zeros((max_word_len, max_word_len), dtype=np.int32)
        srl_rels = np.zeros((max_word_len, max_word_len), dtype=np.int32)
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

        srl_heads = torch.from_numpy(srl_heads).long().to_sparse()
        srl_rels = torch.from_numpy(srl_rels).long().to_sparse()

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
            logger.info("heads:\n{}".format(heads))
            logger.info("rels:\n{}".format(rels))

        features.append(
            InputParsedSrlEnd2EndFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                          attention_mask=tokenized_inputs['attention_mask'][ex_index],
                          token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                          predicate_mask=predicate_mask,
                          srl_heads=srl_heads,
                          srl_rels=srl_rels,
                          word_mask=word_masks[ex_index],
                          first_ids=first_ids_list[ex_index],
                          heads=heads[ex_index] if heads.is_sparse else heads[ex_index].to_sparse(),
                          rels=rels[ex_index] if rels.is_sparse else rels[ex_index].to_sparse()))

    return features



def load_and_cache_examples(args, task, tokenizer, data_type='train', return_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = ReProcessor(task)

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
        #parser_info = "gold-syntax"
        if args.parser_return_tensor:
            parser_info += "-3d"
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
            if args.official_syntax_type == "sdp":
                converter = convert_parsed_examples_to_features_sdp
            else:
                converter = convert_parsed_examples_to_features
            features = converter(examples,
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
    all_srl_heads = torch.stack([f.srl_heads for f in features])
    all_srl_rels = torch.stack([f.srl_rels for f in features])

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
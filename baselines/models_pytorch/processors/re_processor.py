# -*- coding: utf-8 -*-

import logging
import os
import torch
import numpy as np
import pkuseg
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
    if num_items == 6:
        all_input_ids, all_attention_mask, all_token_type_ids, all_ent1_ids, all_ent2_ids, \
        all_labels = map(torch.stack, zip(*batch))
    elif num_items == 8:
        all_input_ids, all_attention_mask, all_token_type_ids, all_ent1_ids, all_ent2_ids, \
        all_labels, all_heads, all_rels = map(torch.stack, zip(*batch))
    elif num_items == 10:
        all_input_ids, all_attention_mask, all_token_type_ids, all_ent1_ids, all_ent2_ids, \
        all_labels, all_heads, all_rels, all_extra_heads, all_extra_rels = map(torch.stack, zip(*batch))
    max_len = max(all_attention_mask.sum(-1)).item()
    # save all_input_ids to decise between word/subword-level 
    all_input_ids_ = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]

    if all_heads is not None:
        head_max_len = max_len
        #print ("head_max_len=",head_max_len)
        if all_heads.is_sparse:
            all_heads = all_heads.to_dense()
            all_rels = all_rels.to_dense()
        assert len(all_heads.size()) == 3
        
        all_heads = all_heads[:, :head_max_len, :head_max_len]
        all_rels = all_rels[:, :head_max_len, :head_max_len]

    if all_extra_heads is not None:
        head_max_len = max_len
        if all_extra_heads.is_sparse:
            all_extra_heads = all_extra_heads.to_dense()
            all_extra_rels = all_extra_rels.to_dense()
        assert len(all_extra_heads.size()) == 3
        
        all_extra_heads = all_extra_heads[:, :head_max_len, :head_max_len]
        all_extra_rels = all_extra_rels[:, :head_max_len, :head_max_len]

    batch = {}
    batch["input_ids"] = all_input_ids_
    batch["attention_mask"] = all_attention_mask
    batch["token_type_ids"] = all_token_type_ids
    batch["ent1_ids"] = all_ent1_ids
    batch["ent2_ids"] = all_ent2_ids
    batch["labels"] = all_labels
    batch["heads"] = all_heads
    batch["rels"] = all_rels
    batch["extra_heads"] = all_extra_heads
    batch["extra_rels"] = all_extra_rels
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


class InputSemSynReFeatures(object):
    """A single set of features of data."""
    def __init__(
            self, 
            input_ids, 
            attention_mask, 
            token_type_ids, 
            heads,
            rels,
            extra_heads,
            extra_rels,
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
        self.heads = heads
        self.rels = rels
        self.extra_heads = extra_heads
        self.extra_rels = extra_rels


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
                InputReExample(guid=guid, text=text, ent1=ent1, beg1=int(beg1), end1=int(end1), 
                                ent2=ent2, beg2=int(beg2), end2=int(end2), label=label))
            #examples[-1].show()
        return examples

def add_entity_identifier(
        examples,
        debug=False
    ):
    segmenter = pkuseg.pkuseg()

    ent1_ids = []
    ent2_ids = []
    texts = []
    for example in examples:
        beg1 = example.beg1
        end1 = example.end1
        beg2 = example.beg2
        end2 = example.end2
        text = example.text
        # ent1 is on left of ent2
        if beg2 >= end1:
            new_text = text[:beg1]+'@'+text[beg1:end1]+'@'+text[end1:beg2]+'#'+text[beg2:end2]+'#'+text[end2:]
            ent1_id = beg1
            ent2_id = beg2 + 2
        # ent1 is on right of ent2
        elif beg1 >= end2:
            new_text = text[:beg2]+'#'+text[beg2:end2]+'#'+text[end2:beg1]+'@'+text[beg1:end1]+'@'+text[end1:]
            ent1_id = beg1 + 2
            ent2_id = beg2
        # overlap
        else:
            if beg1 <= beg2:
                new_text = text[:beg1]+'@'+text[beg1:beg2]+'#'
                ent1_id = beg1
                ent2_id = beg2 + 1
                if end1 <= end2: # only partial overlap
                    new_text += text[beg2:end1]+'@'+text[end1:end2]+'#'+text[end2:]
                else: # end1 > end2, ent2 within ent1
                    new_text += text[beg2:end2]+'#'+text[end2:end1]+'@'+text[end1:]
            else: # beg1 > beg2
                new_text = text[:beg2]+'#'+text[beg2:beg1]+'@'
                ent2_id = beg2
                ent1_id = beg1 + 1
                if end1 <= end2: # ent1 within ent2
                    new_text += text[beg1:end1]+'@'+text[end1:end2]+'#'+text[end2:]
                else: # end1 > end2, only partial overlap
                    new_text += text[beg1:end2]+'#'+text[end2:end1]+'@'+text[end1:]
        if debug:
            #print ("ent1: {}, ent2: {}, old text: {}".format(example.ent1, example.ent2, text))
            example.show()
            print ("new text:", new_text)
            print ("ent1_id={}, ent2_id={}".format(ent1_id, ent2_id))
        
        assert new_text[ent1_id] == '@'
        assert new_text[ent2_id] == '#'
        ent1_ids.append(ent1_id)
        ent2_ids.append(ent2_id)
        texts.append(segmenter.cut(new_text))

    return ent1_ids, ent2_ids, texts


def convert_examples_to_features(
        examples, 
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        is_word_level=False,
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

    ent1_ids, ent2_ids, texts = add_entity_identifier(examples)

    tokenized_inputs = tokenizer(
            texts,
            padding='max_length',
            max_length=max_length,
            is_split_into_words=True,
            return_token_type_ids=True)

    ent1_tok_id = tokenizer.convert_tokens_to_ids('@')
    ent2_tok_id = tokenizer.convert_tokens_to_ids('#')

    features = []

    #word_masks, first_ids_list, word_lens = prepare_word_level_input(
    #    attention_mask=[mask for mask in tokenized_inputs['attention_mask']],
    #    word_ids=[tokenized_inputs.word_ids(i) for i in range(len(examples))],
    #    tokens=[example.words for example in examples])
    #max_word_len = max(word_lens)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        input_ids = tokenized_inputs['input_ids'][ex_index]
        try:
            ent1_id = input_ids.index(ent1_tok_id)
        except:
            logger.info("Cannot find ent1 @ (%d):\n%s" % (ent1_tok_id, " ".join([str(x) for x in input_ids])))
            example.show()
            exit()
        try:
            ent2_id = input_ids.index(ent2_tok_id)
        except:
            logger.info("Cannot find ent2 # (%d):\n%s" % (ent2_tok_id, " ".join([str(x) for x in input_ids])))
            example.show()
            exit()
        label_id = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            #logger.info("guid: %s" % (example.guid))
            #logger.info("sid: %s" % (example.sid))
            example.show()
            logger.info("tokens: %s" % " ".join(texts[ex_index]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in tokenized_inputs['input_ids'][ex_index]]))
            logger.info("ent1={}, ent2={}, label={} ({})".format(ent1_id, ent2_id, label_id, example.label))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in tokenized_inputs['attention_mask'][ex_index]]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in tokenized_inputs['token_type_ids'][ex_index]]))

        features.append(
            InputReFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                          attention_mask=tokenized_inputs['attention_mask'][ex_index],
                          token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                          ent1_id=ent1_id,
                          ent2_id=ent2_id,
                          label=label_id))

    return features


def convert_parsed_examples_to_features(
        examples, 
        tokenizer,
        parser,
        max_length=512,
        task=None,
        label_list=None,
        is_word_level=False,
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

    ent1_ids, ent2_ids, texts = add_entity_identifier(examples)

    tokenized_inputs = tokenizer(
            texts,
            padding='max_length',
            max_length=max_length,
            is_split_into_words=True,
            return_token_type_ids=True)

    word_ids = [tokenized_inputs.word_ids(i) for i in range(len(examples))]

    ent1_tok_id = tokenizer.convert_tokens_to_ids('@')
    ent2_tok_id = tokenizer.convert_tokens_to_ids('#')

    use_multi_parsers = True if isinstance(parser, list) else False

    if use_multi_parsers:
        parsers = parser
        parser = parsers[0]

    heads, rels = parser.add_root_and_parse(
                    tokens=texts,
                    input_ids=tokenized_inputs['input_ids'],
                    attention_mask=tokenized_inputs['attention_mask'],
                    word_ids=word_ids,
                    max_length=max_length,
                    expand_type=expand_type
                )
    if use_multi_parsers:
        # only support 2 different parsers now !
        extra_heads, extra_rels = parsers[1].add_root_and_parse(
                    tokens=texts,
                    input_ids=tokenized_inputs['input_ids'],
                    attention_mask=tokenized_inputs['attention_mask'],
                    word_ids=word_ids,
                    max_length=max_length,
                    expand_type=expand_type
                )

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        
        input_ids = tokenized_inputs['input_ids'][ex_index]
        try:
            ent1_id = input_ids.index(ent1_tok_id)
        except:
            logger.info("Cannot find ent1 @ (%d):\n%s" % (ent1_tok_id, " ".join([str(x) for x in input_ids])))
            example.show()
            exit()
        try:
            ent2_id = input_ids.index(ent2_tok_id)
        except:
            logger.info("Cannot find ent2 # (%d):\n%s" % (ent2_tok_id, " ".join([str(x) for x in input_ids])))
            example.show()
            exit()
        label_id = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            example.show()
            logger.info("input_ids: %s" % " ".join([str(x) for x in tokenized_inputs['input_ids'][ex_index]]))
            logger.info("ent1={}, ent2={}, label={} ({})".format(ent1_id, ent2_id, label_id, example.label))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in tokenized_inputs['attention_mask'][ex_index]]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in tokenized_inputs['token_type_ids'][ex_index]]))
            logger.info("heads:\n{}".format(heads[ex_index]))
            logger.info("rels:\n{}".format(rels[ex_index]))

        if not use_multi_parsers:
            features.append(
                InputParsedReFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                              attention_mask=tokenized_inputs['attention_mask'][ex_index],
                              token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                              ent1_id=ent1_id,
                              ent2_id=ent2_id,
                              label=label_id,
                              heads=heads[ex_index] if heads.is_sparse else heads[ex_index].to_sparse(),
                              rels=rels[ex_index] if rels.is_sparse else rels[ex_index].to_sparse()))
        else:
            features.append(
                InputSemSynReFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                              attention_mask=tokenized_inputs['attention_mask'][ex_index],
                              token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                              ent1_id=ent1_id,
                              ent2_id=ent2_id,
                              label=label_id,
                              heads=heads[ex_index] if heads.is_sparse else heads[ex_index].to_sparse(),
                              rels=rels[ex_index] if rels.is_sparse else rels[ex_index].to_sparse(),
                              extra_heads=extra_heads[ex_index] if extra_heads.is_sparse else extra_heads[ex_index].to_sparse(),
                              extra_rels=extra_rels[ex_index] if extra_rels.is_sparse else extra_rels[ex_index].to_sparse()))

    return features


def load_and_cache_examples(args, task, tokenizer, data_type='train', return_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # be true if parser_model is a list
    use_multi_parsers = isinstance(args.parser_model, list)

    processor = ReProcessor(task)

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
        if not use_multi_parsers:
            cached_features_file = cached_features_filename(args, task if not args.is_word_level else task+"-word-level", data_type=data_type)
        else:
            parser_info = []
            for parser_model in args.parser_model:
                parser_info.append(os.path.basename(parser_model))
            parser_info = "@".join(parser_info)
            cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_parsed_{}_{}_{}'.format(
                data_type,
                list(filter(None, args.model_name_or_path.split('/'))).pop(),
                str(args.max_seq_length),
                str(task),
                parser_info,
                args.parser_expand_type,
                args.parser_align_type))

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
            # this is for single parser_model
            if not isinstance(args.parser_model, list):
                if args.parser_type == "dp":
                    biaffine_parser = Parser(args.parser_model, pretrained_lm="roberta", lm_path=args.parser_lm_path,
                                        batch_size=args.parser_batch, parser_type=args.parser_type)
                elif args.parser_type == "sdp":
                    biaffine_parser = SDPParser(args.parser_model, pretrained_lm="roberta", lm_path=args.parser_lm_path,
                                        batch_size=args.parser_batch, parser_type=args.parser_type)
            # this is for multiple parser_model
            else:
                parsers = []
                for parser_type, parser_model in zip(args.parser_type, args.parser_model):
                    if parser_type == "dp":
                        biaffine_parser = Parser(parser_model, pretrained_lm="roberta", lm_path=args.parser_lm_path,
                                            batch_size=args.parser_batch, parser_type=parser_type)
                    elif parser_type == "sdp":
                        biaffine_parser = SDPParser(parser_model, pretrained_lm="roberta", lm_path=args.parser_lm_path,
                                            batch_size=args.parser_batch, parser_type=parser_type)
                    parsers.append(biaffine_parser)
                biaffine_parser = parsers
                if len(biaffine_parser) > 2:
                    logger.info("Currently only support two different parsers!")
                    exit()
        else:
            biaffine_parser = None

        if biaffine_parser is None:
            features = convert_examples_to_features(examples,
                                            tokenizer,
                                            task=args.task_name,
                                            label_list=label_list,
                                            is_word_level=args.is_word_level,
                                            max_length=args.max_seq_length,
                                            pad_on_left=bool(args.model_type in ['xlnet']),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )
        else:
            features = convert_parsed_examples_to_features(
                                examples,
                                tokenizer,
                                biaffine_parser,
                                task=args.task_name,
                                label_list=label_list,
                                is_word_level=args.is_word_level,
                                max_length=args.max_seq_length,
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
    all_ent1_ids = torch.tensor([f.ent1_id for f in features], dtype=torch.long)
    all_ent2_ids = torch.tensor([f.ent2_id for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    if args.parser_model is None:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_ent1_ids, 
                                all_ent2_ids, all_labels)
    else:
        all_heads = torch.stack([f.heads for f in features])
        all_rels = torch.stack([f.rels for f in features])
        if not use_multi_parsers:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_ent1_ids,
                                    all_ent2_ids, all_labels, all_heads, all_rels)
        else:
            all_extra_heads = torch.stack([f.extra_heads for f in features])
            all_extra_rels = torch.stack([f.extra_rels for f in features])
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_ent1_ids,
                                    all_ent2_ids, all_labels, all_heads, all_rels, all_extra_heads, all_extra_rels)

    if return_examples:
        return dataset, examples
    return dataset
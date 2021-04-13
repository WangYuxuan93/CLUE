# -*- coding: utf-8 -*-

import logging
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from .utils import DataProcessor
from .common import conll09_label_mapping
from processors.processor import cached_features_filename

logger = logging.getLogger(__name__)

class InputSRLExample(object):
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


class InputPredRelFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, predicate_mask, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.predicate_mask = predicate_mask
        self.label_ids = label_ids
        

class InputParsedPredRelFeatures(object):
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


def collate_fn(batch):
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


class SrlProcessor(DataProcessor):
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
        self.label_map = conll09_label_mapping
        return list(self.label_map.keys())

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
        examples = []
        for (i, sent) in enumerate(sents):
            sid = "%s-%s" % (set_type, i)
            pred_ids, _ = self.get_pred_ids(sent)
            tokens_a = [line[1] for line in sent]
            if use_pos:
                pos_tags = [line[5] for line in sent]
            else:
                pos_tags = None
            heads = [line[8] for line in sent]
            rels = [line[10] for line in sent]
            for j, pred_id in enumerate(pred_ids): # the j-th predicate
                guid = "%s-%s" % (set_type, len(examples))
                tokens_b = [tokens_a[pred_id]]
                labels = [line[14+j] if line[14+j] != '_' else 'O' for line in sent]
                examples.append(
                    InputSRLExample(guid=guid, sid=sid, tokens_a=tokens_a, pred_id=pred_id, 
                                    tokens_b=tokens_b, labels=labels, pos_tags=pos_tags,
                                    syntax_heads=heads, syntax_rels=rels))
        return examples


def convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
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
    if task is not None:
        processor = SrlProcessor()
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
            InputPredRelFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
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
    if task is not None:
        processor = SrlProcessor()
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
    for i, example in enumerate(examples):
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
            print ("\nheads:\n", heads[i])
            print ("\nrels:\n", rels[i])
            if dists:
                print ("\ndists:\n", dists[i])

        if compute_dist:
            features.append(
                InputParsedPredRelFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                              attention_mask=tokenized_inputs['attention_mask'][ex_index],
                              token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                              predicate_mask=predicate_mask,
                              label_ids=label_ids,
                              heads=heads[i] if heads.is_sparse else heads[i].to_sparse(),
                              rels=rels[i] if rels.is_sparse else rels[i].to_sparse(),
                              dists=dists[i] if dists.is_sparse else dists[i].to_sparse()))
        else:
            features.append(
                InputParsedPredRelFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                              attention_mask=tokenized_inputs['attention_mask'][ex_index],
                              token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                              predicate_mask=predicate_mask,
                              label_ids=label_ids,
                              heads=heads[i] if heads.is_sparse else heads[i].to_sparse(),
                              rels=rels[i] if rels.is_sparse else rels[i].to_sparse()))
    return features


def floyd(heads, max_len):
    INF = 1e8
    inf = torch.ones_like(heads, device=heads.device, dtype=heads.dtype) * INF
    # replace 0 with infinite
    dist = torch.where(heads==0, inf.long(), heads.long())
    for k in range(max_len):
        for i in range(max_len):
            for j in range(max_len):
                if dist[i][k] != INF and dist[k][j] != INF and dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    zero = torch.zeros_like(heads, device=heads.device).long()
    dist = torch.where(dist==INF, zero, dist)
    return dist

def compute_distance(heads, mask, debug=False):
    if debug:
        torch.set_printoptions(profile="full")

    lengths = [sum(m) for m in mask]
    dists = []
    logger.info("Start computing distance ...")
    # for each sentence
    for i in range(len(heads)):
        if i % 1 == 0:
            print ("%d..."%i, end="")
        if debug:
            print ("heads:\n", heads[i])
            print ("mask:\n", mask[i])
            print ("lengths:\n", lengths[i])
        dist = floyd(heads[i], lengths[i])
        dists.append(dist)
        if debug:
            print ("dist:\n", dist)
    return dists


def load_and_cache_examples(args, task, tokenizer, data_type='train', return_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = SrlProcessor()
    output_mode = 'srl'

    cached_examples_file = os.path.join(args.data_dir, 'cached_examples_{}_{}'.format(data_type, str(task)))
    if return_examples:
        if not os.path.exists(cached_examples_file):
            if data_type == 'train':
                examples = processor.get_train_examples(args.data_dir)
            elif data_type == 'dev':
                examples = processor.get_dev_examples(args.data_dir)
            else:
                examples = processor.get_test_examples(args.data_dir)
            if args.local_rank in [-1, 0]:
                logger.info("Saving examples into cached file %s", cached_examples_file)
                torch.save(examples, cached_examples_file)
        else:
            examples = torch.load(cached_examples_file)
    # Load data features from cache or dataset file
    cached_features_file = cached_features_filename(args, task, data_type=data_type)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        if args.parser_model is not None:
            if args.parser_type == "dp":
                biaffine_parser = Parser(args.parser_model, pretrained_lm="roberta", lm_path=args.parser_lm_path,
                                    batch_size=args.parser_batch, parser_type=args.parser_type)
            elif args.parser_type == "sdp":
                biaffine_parser = SDPParser(args.parser_model, pretrained_lm="roberta", lm_path=args.parser_lm_path,
                                    batch_size=args.parser_batch, parser_type=args.parser_type)
        else:
            biaffine_parser = None

        if biaffine_parser is None:
            features = convert_examples_to_features(examples,
                                                    tokenizer,
                                                    label_list=label_list,
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
                                                    label_list=label_list,
                                                    max_length=args.max_seq_length,
                                                    output_mode=output_mode,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
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
    all_labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    if args.parser_model is None:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_predicate_mask, all_labels)
    else:
        all_heads = torch.stack([f.heads for f in features])
        all_rels = torch.stack([f.rels for f in features])
        if args.parser_compute_dist:
            all_dists = torch.stack([f.dists for f in features])
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_predicate_mask, all_labels, 
                                    all_heads, all_rels, all_dists)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_predicate_mask, all_labels, 
                                    all_heads, all_rels)

    if return_examples:
        return dataset, examples
    return dataset
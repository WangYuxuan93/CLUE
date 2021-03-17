# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import logging
import os
import pickle
import random

import numpy as np
import torch
#from google_albert_pytorch_modeling import AlbertConfig, AlbertForMultipleChoice
#from pytorch_modeling import BertConfig, BertForMultipleChoice, ALBertConfig, ALBertForMultipleChoice
from tools import official_tokenization as tokenization
from tools import utils
from tools.pytorch_optimization import get_optimization, warmup_linear
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import (BertConfig, BertForMultipleChoice, BertTokenizer)
from tools.train_utils import delete_old_checkpoints, load_labels_from_json
from neuronlp2.parser import Parser
from neuronlp2.sdp_parser import SDPParser
from transformers import StructuredBertV2Config, StructuredBertV2ForMultipleChoice

n_class = 4
reverse_order = False
sa_step = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

class InputParsedFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, heads, rels, dists=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.heads = heads
        self.rels = rels
        self.dists = dists

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class c3Processor(DataProcessor):
    def __init__(self, data_dir):
        self.D = [[], [], []]
        self.data_dir = data_dir

        for sid in range(3):
            data = []
            for subtask in ["d", "m"]:
                with open(self.data_dir + "/c3-" + subtask + "-" + ["train.json", "dev.json", "test.json"][sid],
                          "r", encoding="utf8") as f:
                    data += json.load(f)
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                    for k in range(len(data[i][1][j]["choice"])):
                        d += [data[i][1][j]["choice"][k].lower()]
                    for k in range(len(data[i][1][j]["choice"]), 4):
                        d += ['无效答案']  # 有些C3数据选项不足4个，添加[无效答案]能够有效增强模型收敛稳定性
                    if sid in [0, 1]:
                        d += [data[i][1][j]["answer"].lower()]
                    else:
                        # for test set we pick the last choice as answer
                        d += [d[-1]]
                    self.D[sid] += [d]

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self.D[0], "train")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(self.D[2], "test")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        cache_dir = os.path.join(self.data_dir, set_type + '_examples.pkl')
        if os.path.exists(cache_dir):
            examples = pickle.load(open(cache_dir, 'rb'))
        else:
            examples = []
            for (i, d) in enumerate(data):
                answer = -1
                # 这里data[i]有6个元素，0是context，1是问题，2~5是choice，6是答案
                for k in range(4):
                    if data[i][2 + k] == data[i][6]:
                        answer = str(k)

                label = tokenization.convert_to_unicode(answer)

                for k in range(4):
                    guid = "%s-%s-%s" % (set_type, i, k)
                    text_a = tokenization.convert_to_unicode(data[i][0])
                    text_b = tokenization.convert_to_unicode(data[i][k + 2])
                    text_c = tokenization.convert_to_unicode(data[i][1])
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, text_c=text_c))

            with open(cache_dir, 'wb') as w:
                pickle.dump(examples, w)

        return examples

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

def convert_parsed_examples_to_features(
    examples,
    label_list, 
    max_seq_length, 
    tokenizer,
    parser,
    expand_type="word",
    align_type="nltk",
    return_tensor=True,
    compute_dist=False
    ):
    
    print("#examples", len(examples))

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    label_id_list = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)

        tokens_c = tokenizer.tokenize(example.text_c)

        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_c + ["[SEP]"] + tokens_b

        tokens = []
        token_type_ids = []
        tokens.append("[CLS]")
        token_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)
        tokens.append("[SEP]")
        token_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                token_type_ids.append(1)
            tokens.append("[SEP]")
            token_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        label_id = label_map[example.label]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        token_type_ids_list.append(token_type_ids)
        label_id_list.append(label_id)
        

    heads, rels = parser.parse_bpes(
                input_ids_list,
                attention_mask_list,
                has_b=examples[0].text_b is not None,
                has_c=examples[0].text_c is not None,
                expand_type=expand_type,
                max_length=max_seq_length, 
                align_type=align_type, 
                return_tensor=return_tensor, 
                sep_token_id=tokenizer.sep_token_id)

    dists = None
    if compute_dist:
        dists = compute_distance(heads, attention_mask_list)

    features = [[]]
    for i, example in enumerate(examples):
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        if compute_dist:
            features[-1].append(
                InputParsedFeatures(input_ids=input_ids_list[i],
                              attention_mask=attention_mask_list[i],
                              token_type_ids=token_type_ids_list[i],
                              label_id=label_id_list[i],
                              heads=heads[i],
                              rels=rels[i],
                              dists=dists[i]))
        else:
            features[-1].append(
                InputParsedFeatures(input_ids=input_ids_list[i],
                              attention_mask=attention_mask_list[i],
                              token_type_ids=token_type_ids_list[i],
                              label_id=label_id_list[i],
                              heads=heads[i],
                              rels=rels[i]))

        if len(features[-1]) == n_class:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = [[]]
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)

        tokens_c = tokenizer.tokenize(example.text_c)

        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_c + ["[SEP]"] + tokens_b

        tokens = []
        token_type_ids = []
        tokens.append("[CLS]")
        token_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)
        tokens.append("[SEP]")
        token_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                token_type_ids.append(1)
            tokens.append("[SEP]")
            token_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features[-1].append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id))
        if len(features[-1]) == n_class:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features

def load_and_cache_examples(args, tokenizer, examples, data_type='train'):

    task = args.task_name

    processor = c3Processor(args.data_dir)
    label_list = processor.get_labels()

    if args.parser_model is None:
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}.pkl'.format(
            data_type,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task)))
    else:
        parser_info = os.path.basename(args.parser_model)
        if args.parser_return_tensor:
            parser_info += "-3d"
        if args.parser_compute_dist:
            parser_info += "-dist"
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_parsed_{}_{}.pkl'.format(
            data_type,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task),
            parser_info,
            args.parser_expand_type))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        #features = torch.load(cached_features_file)
        features = pickle.load(open(cached_features_file, 'rb'))
    else:
        if args.parser_model is not None:
            if args.parser_type == "dp":
                biaffine_parser = Parser(args.parser_model, pretrained_lm="roberta", lm_path=args.parser_lm_path,
                                    batch_size=args.parser_batch, parser_type=args.parser_type)
            elif args.parser_type == "sdp":
                biaffine_parser = SDPParser(args.parser_model, pretrained_lm="roberta", lm_path=args.parser_lm_path,
                                    batch_size=args.parser_batch, parser_type=args.parser_type)
        else:
            biaffine_parser = None

        logger.info("Creating features from dataset file at %s", args.data_dir)
        if biaffine_parser is None:
            features = convert_examples_to_features(examples, 
                                                    label_list, 
                                                    args.max_seq_length, 
                                                    tokenizer)
        else:
            features = convert_parsed_examples_to_features(examples, 
                                                    label_list, 
                                                    args.max_seq_length, 
                                                    tokenizer,
                                                    biaffine_parser,
                                                    expand_type=args.parser_expand_type,
                                                    align_type=args.parser_align_type,
                                                    return_tensor=args.parser_return_tensor,
                                                    compute_dist=args.parser_compute_dist
                                                    )
    
            del biaffine_parser

        with open(cached_features_file, 'wb') as w:
            pickle.dump(features, w)

    input_ids = []
    attention_mask = []
    token_type_ids = []
    label_id = []
    for f in features:
        input_ids.append([])
        attention_mask.append([])
        token_type_ids.append([])
        for i in range(n_class):
            input_ids[-1].append(f[i].input_ids)
            attention_mask[-1].append(f[i].attention_mask)
            token_type_ids[-1].append(f[i].token_type_ids)
        label_id.append(f[0].label_id)

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_label_ids = torch.tensor(label_id, dtype=torch.long)

    if args.parser_model is None:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    else:
        heads = []
        rels = []
        dists = []
        for f in features:
            heads.append([])
            rels.append([])
            if args.parser_compute_dist:
                dists.append([])
            for i in range(n_class):
                heads[-1].append(f[i].heads)
                rels[-1].append(f[i].rels)
        all_heads = torch.stack([torch.stack(tup) for tup in heads])
        all_rels = torch.stack([torch.stack(tup) for tup in rels])
        if args.parser_compute_dist:
            all_dists = torch.stack([torch.stack(tup) for tup in dists])

        if args.parser_compute_dist:
            all_dists = torch.stack([f.dists for f in features])
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids,
                                    all_heads, all_rels, all_dists)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, 
                                    all_heads, all_rels)

    return dataset


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    num_items = len(batch[0])
    all_heads, all_rels, all_dists = None, None, None

    if num_items == 4:
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels = map(torch.stack, zip(*batch))
    elif num_items == 6:
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_heads, all_rels = map(torch.stack, zip(*batch))
    elif num_items == 7:
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_heads, all_rels, all_dists = map(torch.stack, zip(*batch))
    
    """
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    if num_items >= 6:
        all_heads = all_heads[:, :max_len, :max_len]
        all_rels = all_rels[:, :max_len, :max_len]
    if num_items == 7:
        all_dists = all_dists[:, :max_len, :max_len]
    """
    
    batch = {}
    batch["input_ids"] = all_input_ids
    batch["attention_mask"] = all_attention_mask
    batch["token_type_ids"] = all_token_type_ids
    batch["labels"] = all_labels
    batch["heads"] = all_heads
    batch["rels"] = all_rels
    batch["dists"] = all_dists
    return batch

def _prepare_inputs(inputs, device, use_dist=False, debug=False):

    #print ("inputs:\n", inputs)
    
    for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

    #if "first_indices" in inputs:
    #    del inputs["first_indices"]
    if "heads" in inputs and inputs["heads"] is not None:
        if use_dist:
            if "dists" not in inputs:
                raise ValueError("Distance matrix not provided!")
                exit()
            #torch.set_printoptions(profile="full")
            #print ("dists:\n", inputs["dists"])
            one = torch.ones_like(inputs["dists"]).float()
            zero = torch.zeros_like(inputs["dists"]).float()
            ones = torch.where(inputs["dists"]==0,zero,one)
            dists = torch.where(inputs["dists"]==0,one,inputs["dists"].float())
            dists = ones / dists.float()
            inputs["heads"] = dists
            del inputs["dists"]
            #print ("heads:\n", inputs["heads"])
        else:
            inputs["heads"] = inputs["heads"].float()
            del inputs["dists"]
    else:
        del inputs["heads"]
        del inputs["rels"]
        del inputs["dists"]

    return inputs


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default="bert", type=str, help="Model type")
    parser.add_argument("--gpu_ids", default='0', type=str)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name", default='c3', type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    #parser.add_argument("--bert_config_file", default=None, type=str, required=True,
    #                    help="The config json file corresponding to the pre-trained BERT model. \n"
    #                         "This specifies the model architecture.")
    #parser.add_argument("--vocab_file", default=None, type=str, required=True,
    #                    help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## SBERT parameters
    parser.add_argument("--parser_model", default=None, type=str, help="Parser model's path")
    parser.add_argument("--parser_lm_path", default=None, type=str, help="Parser model's pretrained LM path")
    parser.add_argument("--parser_batch", default=32, type=int, help="Batch size for parser")
    parser.add_argument("--parser_type", default="sdp", type=str, choices=["dp","sdp"], help="Type of the parser")
    parser.add_argument("--parser_expand_type", default="copy", type=str, choices=["copy","word"], help="Policy to expand parses")
    parser.add_argument("--parser_align_type", default="jieba", type=str, choices=["jieba","nltk","rule"], help="Policy to align subwords in parser")
    parser.add_argument("--parser_return_tensor", action='store_true', help="Whether parser should return a tensor")
    parser.add_argument("--parser_compute_dist", action='store_true', help="Whether parser should also compute distance matrix")
    parser.add_argument("--parser_use_reverse_label", action='store_true', help="Whether use reversed parser label")

    ## Other parameters
    #parser.add_argument("--init_checkpoint", default='check_points/pretrain_models/albert_xxlarge_google_zh_v1121/pytorch_model.pth',
    #                    type=str,
    #                    help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument('--logging_steps', type=int, default=10, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument("--num_train_epochs", default=8.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--float16', action='store_true', default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=422, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')

    args = parser.parse_args()
    args.setting_file = os.path.join(args.output_dir, args.setting_file)
    args.log_file = os.path.join(args.output_dir, args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.setting_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print('------------ Options -------------')
        for k in args.__dict__:
            v = args.__dict__[k]
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print('%s: %s' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print('------------ End -------------')
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.per_gpu_train_batch_size
    args.eval_batch_size = args.per_gpu_eval_batch_size
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    processor = c3Processor(args.data_dir)
    label_list = processor.get_labels()

    #tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples()
        num_train_steps = int(len(train_examples) / n_class / args.train_batch_size /
                              args.gradient_accumulation_steps * args.num_train_epochs)
    

    if args.parser_model is not None:
        config = StructuredBertV2Config.from_pretrained(
                        args.config_name if args.config_name else args.model_name_or_path)
        #config.num_labels=num_labels

        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.use_reverse_rel = args.parser_use_reverse_label
        label_path = os.path.join(args.parser_model, "alphabets/type.json")
        parser_label2id = load_labels_from_json(label_path)
        parser_label_embed_size = len(parser_label2id)
        if args.parser_use_reverse_label:
            parser_label_embed_size *= 2
        config.num_rel_labels = parser_label_embed_size
        
        model = StructuredBertV2ForMultipleChoice.from_pretrained(
                        args.model_name_or_path, config=config)
    else:
        config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = BertForMultipleChoice.from_pretrained(args.model_name_or_path, config=config)

    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, config.max_position_embeddings))

    """
    if 'albert' in args.bert_config_file:
        if 'google' in args.bert_config_file:
            bert_config = AlbertConfig.from_json_file(args.bert_config_file)
            model = AlbertForMultipleChoice(bert_config, num_choices=n_class)
        else:
            bert_config = ALBertConfig.from_json_file(args.bert_config_file)
            model = ALBertForMultipleChoice(bert_config, num_choices=n_class)
    else:
        bert_config = BertConfig.from_json_file(args.bert_config_file)
        model = BertForMultipleChoice(bert_config, num_choices=n_class)

    if args.init_checkpoint is not None:
        utils.torch_show_all_params(model)
        utils.torch_init_model(model, args.init_checkpoint)
    """
    if args.float16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = get_optimization(model=model,
                                 float16=args.float16,
                                 learning_rate=args.learning_rate,
                                 total_steps=num_train_steps,
                                 schedule=args.schedule,
                                 warmup_rate=args.warmup_proportion,
                                 max_grad_norm=args.clip_norm,
                                 weight_decay_rate=args.weight_decay_rate,
                                 opt_pooler=True)  # multi_choice must update pooler


    global_step = 0
    eval_dataloader = None
    if args.do_eval:
        eval_examples = processor.get_dev_examples()
        eval_data = load_and_cache_examples(args, tokenizer, eval_examples, data_type='eval')
        """
        feature_dir = os.path.join(args.data_dir, 'dev_features{}.pkl'.format(args.max_seq_length))
        if os.path.exists(feature_dir):
            eval_features = pickle.load(open(feature_dir, 'rb'))
        else:
            eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
            with open(feature_dir, 'wb') as w:
                pickle.dump(eval_features, w)

        input_ids = []
        attention_mask = []
        token_type_ids = []
        label_id = []

        for f in eval_features:
            input_ids.append([])
            attention_mask.append([])
            token_type_ids.append([])
            for i in range(n_class):
                input_ids[-1].append(f[i].input_ids)
                attention_mask[-1].append(f[i].attention_mask)
                token_type_ids[-1].append(f[i].token_type_ids)
            label_id.append(f[0].label_id)

        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        all_label_ids = torch.tensor(label_id, dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
        """
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                    collate_fn=collate_fn)

    if args.do_train:
        train_data = load_and_cache_examples(args, tokenizer, train_examples, data_type='train')
        """
        feature_dir = os.path.join(args.data_dir, 'train_features{}.pkl'.format(args.max_seq_length))
        if os.path.exists(feature_dir):
            train_features = pickle.load(open(feature_dir, 'rb'))
        else:
            train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
            with open(feature_dir, 'wb') as w:
                pickle.dump(train_features, w)
        """

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      drop_last=True, collate_fn=collate_fn)
        steps_per_epoch = int(num_train_steps / args.num_train_epochs)

        log_history = []
        trainer_state = {'best_metric':0, 'best_checkpoint':None, 'epoch': int(args.num_train_epochs)}
        for ie in range(int(args.num_train_epochs)):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            with tqdm(total=int(steps_per_epoch), desc='Epoch %d' % (ie + 1)) as pbar:
                for step, batch in enumerate(train_dataloader):
                    inputs = _prepare_inputs(batch, device)
                    #batch = tuple(t.to(device) for t in batch)
                    #input_ids, attention_mask, token_type_ids, label_ids = batch
                    #print ("input_ids:\n", input_ids)
                    #print ("token_type_ids:\n", token_type_ids)
                    #print ("attention_mask:\n", attention_mask)
                    #print ("label_ids:\n", label_ids)

                    outputs = model(**inputs)
                    loss, logits = outputs[:2]
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    tr_loss += loss.item()

                    if args.float16:
                        optimizer.backward(loss)
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    else:
                        loss.backward()

                    #nb_tr_examples += input_ids.size(0)
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()  # We have accumulated enought gradients
                        model.zero_grad()
                        global_step += 1
                        nb_tr_steps += 1
                        pbar.set_postfix({'loss': '{0:1.5f}'.format(tr_loss / (nb_tr_steps + 1e-5))})
                        pbar.update(1)

                        if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                            result = {'global_step': global_step,
                                      'loss': tr_loss / nb_tr_steps}
                            log_history.append(result)

                        if args.do_eval and args.save_steps > 0 and global_step % args.save_steps == 0:
                            model.eval()
                            eval_loss, eval_accuracy = 0, 0
                            nb_eval_steps, nb_eval_examples = 0, 0
                            logits_all = []
                            #for input_ids, attention_mask, token_type_ids, label_ids in tqdm(eval_dataloader):
                            for step, batch in enumerate(eval_dataloader):
                                inputs = _prepare_inputs(batch, device)
                                """
                                input_ids = input_ids.to(device)
                                attention_mask = attention_mask.to(device)
                                token_type_ids = token_type_ids.to(device)
                                label_ids = label_ids.to(device)
                                """

                                with torch.no_grad():
                                    outputs = model(**inputs)
                                    tmp_eval_loss, logits = outputs[:2]

                                logits = logits.detach().cpu().numpy()
                                #label_ids = label_ids.cpu().numpy()
                                label_ids = inputs['labels'].detach().cpu().numpy()
                                for i in range(len(logits)):
                                    logits_all += [logits[i]]

                                tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

                                eval_loss += tmp_eval_loss.mean().item()
                                eval_accuracy += tmp_eval_accuracy

                                nb_eval_examples += inputs["input_ids"].size(0)
                                nb_eval_steps += 1

                            eval_loss = eval_loss / nb_eval_steps
                            eval_accuracy = eval_accuracy / nb_eval_examples

                            result = {'global_step': global_step,
                                      'eval_loss': eval_loss,
                                      'eval_accuracy': eval_accuracy}
                            log_history.append(result)

                            logger.info("***** Eval results (global step=%d) *****" % global_step)
                            for key in sorted(result.keys()):
                                logger.info("  %s = %s", key, str(result[key]))

                            with open(args.log_file, 'a') as aw:
                                aw.write("-------------------global steps:{}-------------------\n".format(global_step))
                                aw.write(str(json.dumps(result, indent=2)) + '\n')

                            if eval_accuracy > trainer_state['best_metric']:
                                # Save model checkpoint
                                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                                trainer_state['best_checkpoint'] = output_dir
                                trainer_state['best_metric'] = eval_accuracy

                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = model.module if hasattr(model,
                                                                        'module') else model  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                                logger.info("Saving model checkpoint to %s", output_dir)
                                tokenizer.save_pretrained(output_dir)
                                #torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                                delete_old_checkpoints(args.output_dir, trainer_state['best_checkpoint'])

        trainer_state['global_step'] = global_step
        trainer_state['log_history'] = log_history
        state_path = os.path.join(args.output_dir, 'trainer_state.json')
        with open(state_path, 'w') as f:
            json.dump(trainer_state, f,indent=4)

        logger.info("Loading best model from: %s", trainer_state['best_checkpoint'])
        model = model.from_pretrained(trainer_state['best_checkpoint'])
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        #model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_best.pt")))
        #torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    model = model.from_pretrained(args.output_dir)
    model = model.to(device)
    #model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        logits_all = []

        #for input_ids, attention_mask, token_type_ids, label_ids in tqdm(eval_dataloader):
            #input_ids = input_ids.to(device)
            #attention_mask = attention_mask.to(device)
            #token_type_ids = token_type_ids.to(device)
            #label_ids = label_ids.to(device)

        for step, batch in enumerate(eval_dataloader):
            inputs = _prepare_inputs(batch, device)
 
            with torch.no_grad():
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            for i in range(len(logits)):
                logits_all += [logits[i]]

            tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += inputs['input_ids'].size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy}

        output_eval_file = os.path.join(args.output_dir, "results_dev.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        output_eval_file = os.path.join(args.output_dir, "logits_dev.txt")
        with open(output_eval_file, "w") as f:
            for i in range(len(logits_all)):
                for j in range(len(logits_all[i])):
                    f.write(str(logits_all[i][j]))
                    if j == len(logits_all[i]) - 1:
                        f.write("\n")
                    else:
                        f.write(" ")

        test_examples = processor.get_test_examples()
        test_data = load_and_cache_examples(args, tokenizer, test_examples, data_type='test')
        """
        feature_dir = os.path.join(args.data_dir, 'test_features{}.pkl'.format(args.max_seq_length))
        if os.path.exists(feature_dir):
            test_features = pickle.load(open(feature_dir, 'rb'))
        else:
            test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
            with open(feature_dir, 'wb') as w:
                pickle.dump(test_features, w)
        """

        logger.info("***** Running testing *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        """
        input_ids = []
        attention_mask = []
        token_type_ids = []
        label_id = []

        for f in test_features:
            input_ids.append([])
            attention_mask.append([])
            token_type_ids.append([])
            for i in range(n_class):
                input_ids[-1].append(f[i].input_ids)
                attention_mask[-1].append(f[i].attention_mask)
                token_type_ids[-1].append(f[i].token_type_ids)
            label_id.append(f[0].label_id)

        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        all_label_ids = torch.tensor(label_id, dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
        """
        if args.local_rank == -1:
            test_sampler = SequentialSampler(test_data)
        else:
            test_sampler = DistributedSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size, 
                                    collate_fn=collate_fn)

        model.eval()
        test_loss, test_accuracy = 0, 0
        nb_test_steps, nb_test_examples = 0, 0
        logits_all = []

        #for input_ids, attention_mask, token_type_ids, label_ids in tqdm(test_dataloader):
        #    input_ids = input_ids.to(device)
        #    attention_mask = attention_mask.to(device)
        #    token_type_ids = token_type_ids.to(device)
        #    label_ids = label_ids.to(device)

        for step, batch in enumerate(test_dataloader):
            inputs = _prepare_inputs(batch, device)
 
            with torch.no_grad():
                outputs = model(**inputs)
                tmp_test_loss, logits = outputs[:2]

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].to('cpu').numpy()
            for i in range(len(logits)):
                logits_all += [logits[i]]

            tmp_test_accuracy = accuracy(logits, label_ids.reshape(-1))

            test_loss += tmp_test_loss.mean().item()
            test_accuracy += tmp_test_accuracy

            nb_test_examples += inputs['input_ids'].size(0)
            nb_test_steps += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_examples

        result = {'test_loss': test_loss,
                  'test_accuracy': test_accuracy}

        output_test_file = os.path.join(args.output_dir, "results_test.txt")
        with open(output_test_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        output_test_file = os.path.join(args.output_dir, "logits_test.txt")
        with open(output_test_file, "w") as f:
            for i in range(len(logits_all)):
                for j in range(len(logits_all[i])):
                    f.write(str(logits_all[i][j]))
                    if j == len(logits_all[i]) - 1:
                        f.write("\n")
                    else:
                        f.write(" ")

        # the test submission order can't be changed
        submission_test = os.path.join(args.output_dir, "submission_test.json")
        test_preds = [int(np.argmax(logits_)) for logits_ in logits_all]
        with open(submission_test, "w") as f:
            json.dump(test_preds, f)


if __name__ == "__main__":
    main()

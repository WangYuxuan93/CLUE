import collections
import os
import pickle
import logging
import torch

import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import TensorDataset
from neuronlp2.parser import Parser
from neuronlp2.sdp_parser import SDPParser
from .processor import compute_distance, cached_features_filename

try:
    import regex as re
except Exception:
    import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

ChidRawResult = collections.namedtuple("ChidRawResult",
                                   ["unique_id", "example_id", "tag", "logit"])

SPIECE_UNDERLINE = '▁'

def chid_collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    num_items = len(batch[0])
    all_heads, all_rels, all_dists = None, None, None

    if num_items == 6:
        all_input_ids, all_attention_mask, all_token_type_ids, all_choice_masks, all_labels, all_example_index = map(torch.stack, zip(*batch))
    elif num_items == 8:
        all_input_ids, all_attention_mask, all_token_type_ids, all_choice_masks, all_labels, all_example_index, all_heads, all_rels = map(torch.stack, zip(*batch))
    elif num_items == 9:
        all_input_ids, all_attention_mask, all_token_type_ids, all_choice_masks, all_labels, all_example_index, all_heads, all_rels, all_dists = map(torch.stack, zip(*batch))
    
    batch = {}
    batch["input_ids"] = all_input_ids
    batch["attention_mask"] = all_attention_mask
    batch["token_type_ids"] = all_token_type_ids
    batch["labels"] = all_labels
    batch["example_indices"] = all_example_index
    batch["heads"] = all_heads
    batch["rels"] = all_rels
    batch["dists"] = all_dists
    return batch


class ChidExample(object):
    def __init__(self,
                 example_id,
                 tag,
                 doc_tokens,
                 options,
                 answer_index=None):
        self.example_id = example_id
        self.tag = tag
        self.doc_tokens = doc_tokens
        self.options = options
        self.answer_index = answer_index

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "tag: %s" % (self.tag)
        s += ", context: %s" % (''.join(self.doc_tokens))
        s += ", options: [%s]" % (", ".join(self.options))
        if self.answer_index is not None:
            s += ", answer: %s" % self.options[self.answer_index]
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_id,
                 tag,
                 tokens,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 choice_masks,
                 label=None):
        self.unique_id = unique_id
        self.example_id = example_id
        self.tag = tag
        self.tokens = tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.choice_masks = choice_masks
        self.label = label  # 正确答案在所有候选答案中的index


class InputParsedFeatures(object):

    def __init__(self,
                 unique_id,
                 example_id,
                 tag,
                 tokens,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 choice_masks,
                 label=None,
                 heads=None,
                 rels=None,
                 dists=None):
        self.unique_id = unique_id
        self.example_id = example_id
        self.tag = tag
        self.tokens = tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.choice_masks = choice_masks
        self.label = label  # 正确答案在所有候选答案中的index
        self.heads = heads
        self.rels = rels
        self.dists = dists


def read_chid_examples(input_data_file, input_label_file, is_training=True):
    '''
    将原始数据处理为如下形式：
    part_passage遍历每个blak的周围位置
    :param input_data:
    :param is_training:
    :return:
    '''

    if is_training:
        input_label = json.load(open(input_label_file))
    input_data = open(input_data_file)

    def _is_chinese_char(cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def is_fuhao(c):
        if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                or c == '‘' or c == '’':
            return True
        return False

    def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        is_blank = False
        for index, char in enumerate(text):
            cp = ord(char)
            if is_blank:
                output.append(char)
                if context[index - 12:index + 1].startswith("#idiom"):
                    is_blank = False
                    output.append(SPIECE_UNDERLINE)
            else:
                if text[index:index + 6] == "#idiom":
                    is_blank = True
                    if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                        output.append(SPIECE_UNDERLINE)
                    output.append(char)
                elif _is_chinese_char(cp) or is_fuhao(char):
                    if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                        output.append(SPIECE_UNDERLINE)
                    output.append(char)
                    output.append(SPIECE_UNDERLINE)
                else:
                    output.append(char)
        return "".join(output)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
            return True
        return False

    examples = []
    example_id = 0
    for data in tqdm(input_data):

        data = eval(data)
        options = data['candidates']

        for context in data['content']:

            context = context.replace("“", "\"").replace("”", "\"").replace("——", "--"). \
                replace("—", "-").replace("―", "-").replace("…", "...").replace("‘", "\'").replace("’", "\'")
            context = _tokenize_chinese_chars(context)

            paragraph_text = context.strip()
            doc_tokens = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False

            tags = [blank for blank in doc_tokens if '#idiom' in blank]

            if is_training:
                for tag_index, tag in enumerate(tags):
                    answer_index = input_label[tag]
                    example = ChidExample(
                        example_id=example_id,
                        tag=tag,
                        doc_tokens=doc_tokens,
                        options=options,
                        answer_index=answer_index)
                    examples.append(example)
            else:
                for tag_index, tag in enumerate(tags):
                    example = ChidExample(
                        example_id=example_id,
                        tag=tag,
                        doc_tokens=doc_tokens,
                        options=options)
                    examples.append(example)
        else:
            example_id += 1
    else:
        print('原始样本个数：{}'.format(example_id))

    print('实际生成总样例数：{}'.format(len(examples)))
    return examples


def add_tokens_for_around(tokens, pos, num_tokens):
    num_l = num_tokens // 2
    num_r = num_tokens - num_l

    if pos >= num_l and (len(tokens) - 1 - pos) >= num_r:
        tokens_l = tokens[pos - num_l: pos]
        tokens_r = tokens[pos + 1: pos + 1 + num_r]
    elif pos <= num_l:
        tokens_l = tokens[:pos]
        right_len = num_tokens - len(tokens_l)
        tokens_r = tokens[pos + 1: pos + 1 + right_len]
    elif (len(tokens) - 1 - pos) <= num_r:
        tokens_r = tokens[pos + 1:]
        left_len = num_tokens - len(tokens_r)
        tokens_l = tokens[pos - left_len: pos]
    else:
        raise ValueError('impossible')

    return tokens_l, tokens_r


def convert_examples_to_features(
        examples, 
        tokenizer, 
        max_seq_length=128, 
        max_num_choices=10
    ):
    '''
    将所有候选答案放置在片段开头
    '''

    def _loop(example, unique_id, label):
        '''
        :param example:
        :param unique_id:
        :return:
            input_ids = (C, seq_len)
            token_type_ids = (C, seq_len) = segment_id
            input_mask = (C, seq_len)
            labels = int
            choices_mask = (C)
        '''
        input_ids = []
        attention_mask = []
        token_type_ids = []
        choice_masks = [1] * len(example.options)

        tag = example.tag
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            if '#idiom' in token:
                sub_tokens = [str(token)]
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                all_doc_tokens.append(sub_token)

        pos = all_doc_tokens.index(tag)
        num_tokens = max_tokens_for_doc - 5  # [unused1]和segA的成语
        tmp_l, tmp_r = add_tokens_for_around(all_doc_tokens, pos, num_tokens)
        num_l = len(tmp_l)
        num_r = len(tmp_r)

        tokens_l = []
        for token in tmp_l:
            if '#idiom' in token and token != tag:
                tokens_l.extend(['[MASK]'] * 4)
            else:
                tokens_l.append(token)
        tokens_l = tokens_l[-num_l:]
        del tmp_l

        tokens_r = []
        for token in tmp_r:
            if '#idiom' in token and token != tag:
                tokens_r.extend(['[MASK]'] * 4)
            else:
                tokens_r.append(token)
        tokens_r = tokens_r[: num_r]
        del tmp_r

        for i, elem in enumerate(example.options):
            option = tokenizer.tokenize(elem)
            tokens = ['[CLS]'] + option + ['[SEP]'] + tokens_l + ['[unused1]'] + tokens_r + ['[SEP]']

            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_id)
            segment_id = [0] * len(input_id)

            while len(input_id) < max_seq_length:
                input_id.append(0)
                input_mask.append(0)
                segment_id.append(0)
            assert len(input_id) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_id) == max_seq_length

            input_ids.append(input_id)
            attention_mask.append(input_mask)
            token_type_ids.append(segment_id)

        if unique_id < 5:
            print("*** Example ***")
            print("unique_id: {}".format(unique_id))
            print("context_id: {}".format(tag))
            print("label: {}".format(label))
            print("tag_index: {}".format(pos))
            print("tokens: {}".format("".join(tokens)))
            print("choice_masks: {}".format(choice_masks))
            print("input_ids:\n", input_ids)
        while len(input_ids) < max_num_choices:
            input_ids.append([0] * max_seq_length)
            attention_mask.append([0] * max_seq_length)
            token_type_ids.append([0] * max_seq_length)
            choice_masks.append(0)
        assert len(input_ids) == max_num_choices
        assert len(attention_mask) == max_num_choices
        assert len(token_type_ids) == max_num_choices
        assert len(choice_masks) == max_num_choices

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_id=example.example_id,
                tag=tag,
                tokens=tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                choice_masks=choice_masks,
                label=label))

    max_tokens_for_doc = max_seq_length - 3  # [CLS] choice [SEP] document [SEP]
    features = []
    unique_id = 0

    for (example_index, example) in enumerate(tqdm(examples)):

        label = example.answer_index
        if label != None:
            _loop(example, unique_id, label)
        else:
            _loop(example, unique_id, None)
        unique_id += 1

        if unique_id % 12000 == 0:
            print("unique_id: %s" % (unique_id))
    print("unique_id: %s" % (unique_id))
    return features


def convert_parsed_examples_to_features(
        examples, 
        tokenizer, 
        parser,
        max_seq_length=128, 
        max_num_choices=10,
        expand_type="word",
        align_type="nltk",
        return_tensor=True,
        compute_dist=False
    ):
    '''
    将所有候选答案放置在片段开头
    '''

    def _loop(example, unique_id, label):
        '''
        :param example:
        :param unique_id:
        :return:
            input_ids = (C, seq_len)
            token_type_ids = (C, seq_len) = segment_id
            input_mask = (C, seq_len)
            labels = int
            choices_mask = (C)
        '''
        input_ids = []
        attention_mask = []
        token_type_ids = []
        choice_masks = [1] * len(example.options)

        tag = example.tag
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            if '#idiom' in token:
                sub_tokens = [str(token)]
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                all_doc_tokens.append(sub_token)

        pos = all_doc_tokens.index(tag)
        num_tokens = max_tokens_for_doc - 5  # [unused1]和segA的成语
        tmp_l, tmp_r = add_tokens_for_around(all_doc_tokens, pos, num_tokens)
        num_l = len(tmp_l)
        num_r = len(tmp_r)

        tokens_l = []
        for token in tmp_l:
            if '#idiom' in token and token != tag:
                tokens_l.extend(['[MASK]'] * 4)
            else:
                tokens_l.append(token)
        tokens_l = tokens_l[-num_l:]
        del tmp_l

        tokens_r = []
        for token in tmp_r:
            if '#idiom' in token and token != tag:
                tokens_r.extend(['[MASK]'] * 4)
            else:
                tokens_r.append(token)
        tokens_r = tokens_r[: num_r]
        del tmp_r

        for i, elem in enumerate(example.options):
            option = tokenizer.tokenize(elem)
            tokens = ['[CLS]'] + option + ['[SEP]'] + tokens_l + ['[unused1]'] + tokens_r + ['[SEP]']

            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_id)
            segment_id = [0] * len(input_id)

            while len(input_id) < max_seq_length:
                input_id.append(0)
                input_mask.append(0)
                segment_id.append(0)
            assert len(input_id) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_id) == max_seq_length

            input_ids.append(input_id)
            attention_mask.append(input_mask)
            token_type_ids.append(segment_id)

        while len(input_ids) < max_num_choices:
            input_ids.append([0] * max_seq_length)
            attention_mask.append([0] * max_seq_length)
            token_type_ids.append([0] * max_seq_length)
            choice_masks.append(0)

        assert len(input_ids) == max_num_choices
        assert len(attention_mask) == max_num_choices
        assert len(token_type_ids) == max_num_choices
        assert len(choice_masks) == max_num_choices

        unique_id_list.append(unique_id)
        example_id_list.append(example.example_id)
        tag_list.append(tag)
        tokens_list.append(tokens)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        token_type_ids_list.append(token_type_ids)
        choice_masks_list.append(choice_masks)
        label_list.append(label)

    # data collector
    unique_id_list = []
    example_id_list = []
    tag_list = []
    tokens_list = []
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    choice_masks_list = []
    label_list = []

    max_tokens_for_doc = max_seq_length - 3  # [CLS] choice [SEP] document [SEP]
    features = []
    unique_id = 0

    for (example_index, example) in enumerate(tqdm(examples)):

        label = example.answer_index
        if label != None:
            _loop(example, unique_id, label)
        else:
            _loop(example, unique_id, None)
        unique_id += 1

        if unique_id % 12000 == 0:
            print("unique_id: %s" % (unique_id))
    print("unique_id: %s" % (unique_id))
    # (num_examples*max_num_choices, max_seq_len)
    flat_input_ids_list = [input_id for input_ids in input_ids_list for input_id in input_ids]
    flat_attention_mask_list = [input_mask for attention_mask in attention_mask_list for input_mask in attention_mask]
    # (num_examples*max_num_choices, seq_len, seq_len)
    heads, rels = parser.parse_bpes(
            flat_input_ids_list,
            flat_attention_mask_list,
            has_b=True,
            expand_type=expand_type,
            max_length=max_seq_length, 
            align_type=align_type, 
            return_tensor=return_tensor, 
            sep_token_id=tokenizer.sep_token_id,
            max_num_choices=max_num_choices)

    #heads = flat_heads.split(max_num_choices, dim=0)
    #rels = flat_rels.split(max_num_choices, dim=0)
    assert len(heads) == len(unique_id_list)
    assert len(rels) == len(unique_id_list)

    dists = None
    if compute_dist:
        flat_dists = compute_distance(heads, attention_mask_list)
        dists = flat_dists.split(max_num_choices, dim=0)
        assert len(dists) == len(unique_id_list)

    for i in range(len(unique_id_list)):
        if unique_id_list[i] < 5:
            torch.set_printoptions(profile="full")
            print("*** Example ***")
            print("unique_id: {}".format(unique_id_list[i]))
            print("context_id: {}".format(tag_list[i]))
            print("label: {}".format(label_list[i]))
            print("tag: {}".format(tag_list[i]))
            print("tokens: {}".format("".join(tokens_list[i])))
            print("choice_masks: {}".format(choice_masks_list[i]))
            print("input_ids:\n", input_ids_list[i])
            print("heads:\n", heads[i])
            print("rels:\n", rels[i])

        if compute_dist:
            features.append(
                InputParsedFeatures(
                    unique_id=unique_id_list[i],
                    example_id=example_id_list[i],
                    tag=tag_list[i],
                    tokens=tokens_list[i],
                    input_ids=input_ids_list[i],
                    attention_mask=attention_mask_list[i],
                    token_type_ids=token_type_ids_list[i],
                    choice_masks=choice_masks_list[i],
                    label=label_list[i],
                    heads=heads[i] if heads[i].is_sparse else heads[i].to_sparse(),
                    rels=rels[i] if rels[i].is_sparse else rels[i].to_sparse(),
                    dists=dists[i] if dists[i].is_sparse else dists[i].to_sparse()))
        else:
            features.append(
                InputParsedFeatures(
                    unique_id=unique_id_list[i],
                    example_id=example_id_list[i],
                    tag=tag_list[i],
                    tokens=tokens_list[i],
                    input_ids=input_ids_list[i],
                    attention_mask=attention_mask_list[i],
                    token_type_ids=token_type_ids_list[i],
                    choice_masks=choice_masks_list[i],
                    label=label_list[i],
                    heads=heads[i] if heads[i].is_sparse else heads[i].to_sparse(),
                    rels=rels[i] if rels[i].is_sparse else rels[i].to_sparse()))

    return features


def logits_matrix_to_array(logits_matrix, index_2_idiom):
    """从矩阵中计算全局概率最大的序列"""
    logits_matrix = np.array(logits_matrix)
    logits_matrix = np.transpose(logits_matrix)
    tmp = []
    for i, row in enumerate(logits_matrix):
        for j, col in enumerate(row):
            tmp.append((i, j, col))
    else:
        choice = set(range(i + 1))
        blanks = set(range(j + 1))
    tmp = sorted(tmp, key=lambda x: x[2], reverse=True)
    results = []
    for i, j, v in tmp:
        if (j in blanks) and (i in choice):
            results.append((i, j))
            blanks.remove(j)
            choice.remove(i)
    results = sorted(results, key=lambda x: x[1], reverse=False)
    results = [[index_2_idiom[j], i] for i, j in results]
    return results


def logits_matrix_max_array(logits_matrix, index_2_idiom):
    logits_matrix = np.array(logits_matrix)
    arg_max = logits_matrix.argmax(axis=1)
    results = [[index_2_idiom[i], idx] for i, idx in enumerate(arg_max)]
    return results


def get_final_predictions(all_results, g=True):
    #if not os.path.exists(tmp_predict_file):
    #    pickle.dump(all_results, open(tmp_predict_file, 'wb'))

    raw_results = {}
    for i, elem in enumerate(all_results):
        example_id = elem.example_id
        if example_id not in raw_results:
            raw_results[example_id] = [(elem.tag, elem.logit)]
        else:
            raw_results[example_id].append((elem.tag, elem.logit))

    results = []
    for example_id, elem in raw_results.items():
        index_2_idiom = {index: tag for index, (tag, logit) in enumerate(elem)}
        logits = [logit for _, logit in elem]
        if g:
            results.extend(logits_matrix_to_array(logits, index_2_idiom))
        else:
            results.extend(logits_matrix_max_array(logits, index_2_idiom))
    return results


def write_predictions(results, output_prediction_file):
    # output_prediction_file = result6.csv
    # results = pd.DataFrame(results)
    # results.to_csv(output_prediction_file, header=None, index=None)

    results_dict = {}
    for result in results:
        results_dict[result[0]] = result[1]
    with open(output_prediction_file, 'w') as w:
        json.dump(results_dict, w, indent=2)

    print("Writing predictions to: {}".format(output_prediction_file))


def generate_input(data_file, label_file, example_file, feature_file, tokenizer, max_seq_length, max_num_choices,
                   is_training=True):
    if os.path.exists(feature_file):
        features = pickle.load(open(feature_file, 'rb'))
    elif os.path.exists(example_file):
        examples = pickle.load(open(example_file, 'rb'))
        features = convert_examples_to_features(examples, tokenizer, max_seq_length, max_num_choices)
        pickle.dump(features, open(feature_file, 'wb'))
    else:
        examples = read_chid_examples(data_file, label_file, is_training=is_training)
        pickle.dump(examples, open(example_file, 'wb'))
        features = convert_examples_to_features(examples, tokenizer, max_seq_length, max_num_choices)
        pickle.dump(features, open(feature_file, 'wb'))

    return features


def load_and_cache_chid_examples(args, task, tokenizer, data_type='train', 
                                 return_examples=False, return_features=False):
    cached_features_file = cached_features_filename(args, task, data_type=data_type)

    data_file = os.path.join(args.data_dir, data_type+'.json')
    label_file = os.path.join(args.data_dir, data_type+'_answer.json')

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        examples = read_chid_examples(data_file, label_file, is_training=True if data_type in ['train','dev'] else False)

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
                                                    args.max_seq_length, 
                                                    args.max_num_choices)
        else:
            features = convert_parsed_examples_to_features(examples, 
                                                    tokenizer,
                                                    biaffine_parser, 
                                                    args.max_seq_length, 
                                                    args.max_num_choices,
                                                    expand_type=args.parser_expand_type,
                                                    align_type=args.parser_align_type,
                                                    return_tensor=args.parser_return_tensor,
                                                    compute_dist=args.parser_compute_dist
                                                    )
    
            del biaffine_parser

        torch.save(features, cached_features_file)


    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_choice_masks = torch.tensor([f.choice_masks for f in features], dtype=torch.long)
    
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    if data_type in ['train','dev']:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        # this will not be used in predict
        all_labels = all_example_index
        #all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    
    if args.parser_model is None:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_choice_masks, 
                                all_labels, all_example_index)
    else:
        all_heads = torch.stack([f.heads for f in features])
        all_rels = torch.stack([f.rels for f in features])
            
        if args.parser_compute_dist:
            all_dists = torch.stack([f.dists for f in features])
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_choice_masks, 
                                    all_labels, all_example_index, all_heads, all_rels, all_dists)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_choice_masks, 
                                    all_labels, all_example_index, all_heads, all_rels)

    outputs = (dataset,)
    if return_features:
        outputs += (features,)
    return outputs


def evaluate(ans_f, pre_f):
    ans = json.load(open(ans_f))
    pre = json.load(open(pre_f))

    total_num = 0
    acc_num = 0
    for id_ in ans:
        if id_ not in pre:
            raise FileNotFoundError
        total_num += 1
        if ans[id_] == pre[id_]:
            acc_num += 1

    acc = acc_num / total_num
    acc *= 100
    return acc


"""
def convert_parsed_examples_to_features(
        examples, 
        tokenizer, 
        parser,
        max_seq_length=128, 
        max_num_choices=10,
        expand_type="word",
        align_type="nltk",
        return_tensor=True,
        compute_dist=False
    ):
    '''
    将所有候选答案放置在片段开头
    '''

    def _loop(example, unique_id, label):
        '''
        :param example:
        :param unique_id:
        :return:
            input_ids = (C, seq_len)
            token_type_ids = (C, seq_len) = segment_id
            input_mask = (C, seq_len)
            labels = int
            choices_mask = (C)
        '''
        input_ids = []
        attention_mask = []
        token_type_ids = []
        choice_masks = [1] * len(example.options)

        tag = example.tag
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            if '#idiom' in token:
                sub_tokens = [str(token)]
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                all_doc_tokens.append(sub_token)

        pos = all_doc_tokens.index(tag)
        num_tokens = max_tokens_for_doc - 5  # [unused1]和segA的成语
        tmp_l, tmp_r = add_tokens_for_around(all_doc_tokens, pos, num_tokens)
        num_l = len(tmp_l)
        num_r = len(tmp_r)

        tokens_l = []
        for token in tmp_l:
            if '#idiom' in token and token != tag:
                tokens_l.extend(['[MASK]'] * 4)
            else:
                tokens_l.append(token)
        tokens_l = tokens_l[-num_l:]
        del tmp_l

        tokens_r = []
        for token in tmp_r:
            if '#idiom' in token and token != tag:
                tokens_r.extend(['[MASK]'] * 4)
            else:
                tokens_r.append(token)
        tokens_r = tokens_r[: num_r]
        del tmp_r

        for i, elem in enumerate(example.options):
            option = tokenizer.tokenize(elem)
            tokens = ['[CLS]'] + option + ['[SEP]'] + tokens_l + ['[unused1]'] + tokens_r + ['[SEP]']

            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_id)
            segment_id = [0] * len(input_id)

            while len(input_id) < max_seq_length:
                input_id.append(0)
                input_mask.append(0)
                segment_id.append(0)
            assert len(input_id) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_id) == max_seq_length

            input_ids.append(input_id)
            attention_mask.append(input_mask)
            token_type_ids.append(segment_id)

        heads, rels = parser.parse_bpes(
            input_ids,
            attention_mask,
            has_b=True,
            expand_type=expand_type,
            max_length=max_seq_length, 
            align_type=align_type, 
            return_tensor=return_tensor, 
            sep_token_id=tokenizer.sep_token_id)

        dists = None
        if compute_dist:
            dists = compute_distance(heads, attention_mask_list)

        if unique_id < 5:
            torch.set_printoptions(profile="full")
            print("*** Example ***")
            print("unique_id: {}".format(unique_id))
            print("context_id: {}".format(tag))
            print("label: {}".format(label))
            print("tag_index: {}".format(pos))
            print("tokens: {}".format("".join(tokens)))
            print("choice_masks: {}".format(choice_masks))
            print("input_ids:\n", input_ids)
            print("heads:\n", heads)
            print("rels:\n", rels)

        while len(input_ids) < max_num_choices:
            input_ids.append([0] * max_seq_length)
            attention_mask.append([0] * max_seq_length)
            token_type_ids.append([0] * max_seq_length)
            choice_masks.append(0)

        assert len(input_ids) == max_num_choices
        assert len(attention_mask) == max_num_choices
        assert len(token_type_ids) == max_num_choices
        assert len(choice_masks) == max_num_choices

        if compute_dist:
            features.append(
                InputParsedFeatures(
                    unique_id=unique_id,
                    example_id=example.example_id,
                    tag=tag,
                    tokens=tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    choice_masks=choice_masks,
                    label=label,
                    heads=heads.to_sparse(),
                    rels=rels.to_sparse(),
                    dists=dists.to_sparse()))
        else:
            features.append(
                InputParsedFeatures(
                    unique_id=unique_id,
                    example_id=example.example_id,
                    tag=tag,
                    tokens=tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    choice_masks=choice_masks,
                    label=label,
                    heads=heads.to_sparse(),
                    rels=rels.to_sparse()))

    max_tokens_for_doc = max_seq_length - 3  # [CLS] choice [SEP] document [SEP]
    features = []
    unique_id = 0

    for (example_index, example) in enumerate(tqdm(examples)):

        label = example.answer_index
        if label != None:
            _loop(example, unique_id, label)
        else:
            _loop(example, unique_id, None)
        unique_id += 1

        if unique_id % 12000 == 0:
            print("unique_id: %s" % (unique_id))
    print("unique_id: %s" % (unique_id))
    return features
"""
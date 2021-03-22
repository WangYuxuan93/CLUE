import collections
import json
import os
import torch

from tqdm import tqdm
from torch.utils.data import TensorDataset
from neuronlp2.parser import Parser
from neuronlp2.sdp_parser import SDPParser
from .utils import get_final_text, _get_best_indexes, _compute_softmax, calc_f1_score, calc_em_score
from collections import OrderedDict

from tools import official_tokenization as tokenization

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

CmrcRawResult = collections.namedtuple("CmrcRawResult",
                                    ["unique_id", "start_logits", "end_logits"])

SPIECE_UNDERLINE = '▁'


def cmrc2018_collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    num_items = len(batch[0])
    all_heads, all_rels, all_dists = None, None, None

    if num_items == 6:
        all_input_ids, all_attention_mask, all_token_type_ids, all_start_positions, all_end_positions, all_example_index = map(torch.stack, zip(*batch))
    elif num_items == 8:
        all_input_ids, all_attention_mask, all_token_type_ids, all_start_positions, all_end_positions, all_example_index, all_heads, all_rels = map(torch.stack, zip(*batch))
    elif num_items == 9:
        all_input_ids, all_attention_mask, all_token_type_ids, all_start_positions, all_end_positions, all_example_index, all_heads, all_rels, all_dists = map(torch.stack, zip(*batch))
    
    batch = {}
    batch["input_ids"] = all_input_ids
    batch["attention_mask"] = all_attention_mask
    batch["token_type_ids"] = all_token_type_ids
    batch["start_positions"] = all_start_positions
    batch["end_positions"] = all_end_positions
    batch["example_indices"] = all_example_index
    batch["heads"] = all_heads
    batch["rels"] = all_rels
    batch["dists"] = all_dists
    return batch


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def read_cmrc2018_examples(input_data_file, repeat_limit=3, is_training=True):
    with open(input_data_file, 'r') as f:
        train_data = json.load(f)
        train_data = train_data['data']

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
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp) or is_fuhao(char):
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

    # to examples
    examples = []
    mis_match = 0
    for article in tqdm(train_data):
        for para in article['paragraphs']:
            context = para['context']
            context_chs = _tokenize_chinese_chars(context)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in context_chs:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                if c != SPIECE_UNDERLINE:
                    char_to_word_offset.append(len(doc_tokens) - 1)

            for qas in para['qas']:
                qid = qas['id']
                ques_text = qas['question']
                ans_text = qas['answers'][0]['text']

                start_position_final = None
                end_position_final = None
                if is_training:
                    count_i = 0
                    start_position = qas['answers'][0]['answer_start']

                    end_position = start_position + len(ans_text) - 1
                    while context[start_position:end_position + 1] != ans_text and count_i < repeat_limit:
                        start_position -= 1
                        end_position -= 1
                        count_i += 1

                    while context[start_position] == " " or context[start_position] == "\t" or \
                            context[start_position] == "\r" or context[start_position] == "\n":
                        start_position += 1

                    start_position_final = char_to_word_offset[start_position]
                    end_position_final = char_to_word_offset[end_position]

                    if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
                        start_position_final += 1

                    actual_text = "".join(doc_tokens[start_position_final:(end_position_final + 1)])
                    cleaned_answer_text = "".join(tokenization.whitespace_tokenize(ans_text))

                    if actual_text != cleaned_answer_text:
                        print(actual_text, 'V.S', cleaned_answer_text)
                        mis_match += 1
                        # ipdb.set_trace()

                examples.append({'doc_tokens': doc_tokens,
                                 'orig_answer_text': ans_text,
                                 'qid': qid,
                                 'question': ques_text,
                                 'answer': ans_text,
                                 'start_position': start_position_final,
                                 'end_position': end_position_final})

    print('examples num:', len(examples))
    print('mis_match:', mis_match)
    #os.makedirs('/'.join(output_files[0].split('/')[0:-1]), exist_ok=True)
    #json.dump(examples, open(output_files[0], 'w'))

    return examples


def convert_examples_to_features(
        examples, 
        tokenizer, 
        is_training=True,
        max_query_length=64,
        max_seq_length=128,
        doc_stride=128
    ):
    # to features
    features = []
    unique_id = 1000000000
    for (example_index, example) in enumerate(tqdm(examples)):
        query_tokens = tokenizer.tokenize(example['question'])
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example['doc_tokens']):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
            if example['end_position'] < len(example['doc_tokens']) - 1:
                tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example['orig_answer_text'])

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        doc_spans = []
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            token_type_ids = []
            tokens.append("[CLS]")
            token_type_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                token_type_ids.append(0)
            tokens.append("[SEP]")
            token_type_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
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

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                if tok_start_position == -1 and tok_end_position == -1:
                    start_position = 0  # 问题本来没答案，0是[CLS]的位子
                    end_position = 0
                else:  # 如果原本是有答案的，那么去除没有答案的feature
                    out_of_span = False
                    doc_start = doc_span.start  # 映射回原文的起点和终点
                    doc_end = doc_span.start + doc_span.length - 1

                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

            if unique_id < 1000000005:
                torch.set_printoptions(profile="full")
                print("*** Example ***")
                print("unique_id: {}".format(unique_id))
                print("example_index: {}".format(example_index))
                print("doc_span_index: {}".format(doc_span_index))
                print("tokens: {}".format("".join(tokens)))
                print("token_to_orig_map: {}".format(token_to_orig_map))
                print("token_is_max_context: {}".format(token_is_max_context))
                print("input_ids:\n", input_ids)
                print("start_position:\n", start_position)
                print("end_position:\n", end_position)

            features.append({'unique_id': unique_id,
                             'example_index': example_index,
                             'doc_span_index': doc_span_index,
                             'tokens': tokens,
                             'token_to_orig_map': token_to_orig_map,
                             'token_is_max_context': token_is_max_context,
                             'input_ids': input_ids,
                             'attention_mask': attention_mask,
                             'token_type_ids': token_type_ids,
                             'start_position': start_position,
                             'end_position': end_position})
            unique_id += 1

    #print('features num:', len(features))
    #json.dump(features, open(output_files[1], 'w'))

    return features


def convert_parsed_examples_to_features(
        examples, 
        tokenizer,
        parser,
        is_training=True,
        max_query_length=64,
        max_seq_length=128,
        doc_stride=128,
        expand_type="word",
        align_type="nltk",
        return_tensor=True,
        compute_dist=False
    ):
    # to features
    features = []
    unique_id = 1000000000
    for (example_index, example) in enumerate(tqdm(examples)):
        query_tokens = tokenizer.tokenize(example['question'])
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example['doc_tokens']):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
            if example['end_position'] < len(example['doc_tokens']) - 1:
                tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example['orig_answer_text'])

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        doc_spans = []
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        # data collector
        unique_id_list = []
        doc_span_index_list = []
        tokens_list = []
        token_to_orig_map_list = []
        token_is_max_context_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        start_position_list = []
        end_position_list = []

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            token_type_ids = []
            tokens.append("[CLS]")
            token_type_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                token_type_ids.append(0)
            tokens.append("[SEP]")
            token_type_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
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

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                if tok_start_position == -1 and tok_end_position == -1:
                    start_position = 0  # 问题本来没答案，0是[CLS]的位子
                    end_position = 0
                else:  # 如果原本是有答案的，那么去除没有答案的feature
                    out_of_span = False
                    doc_start = doc_span.start  # 映射回原文的起点和终点
                    doc_end = doc_span.start + doc_span.length - 1

                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

            unique_id_list.append(unique_id)
            doc_span_index_list.append(doc_span_index)
            tokens_list.append(tokens)
            token_to_orig_map_list.append(token_to_orig_map)
            token_is_max_context_list.append(token_is_max_context)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            token_type_ids_list.append(token_type_ids)
            start_position_list.append(start_position)
            end_position_list.append(end_position)

            unique_id += 1

        heads, rels = parser.parse_bpes(
            input_ids_list,
            attention_mask_list,
            has_b=True,
            expand_type=expand_type,
            max_length=max_seq_length, 
            align_type=align_type, 
            return_tensor=return_tensor, 
            sep_token_id=tokenizer.sep_token_id)

        dists = None
        if compute_dist:
            dists = compute_distance(heads, attention_mask_list)

        for (i, doc_span) in enumerate(doc_spans):
            doc_span_index = i
            if unique_id_list[i] < 1000000005:
                torch.set_printoptions(profile="full")
                print("*** Example ***")
                print("unique_id: {}".format(unique_id_list[i]))
                print("example_index: {}".format(example_index))
                print("doc_span_index: {}".format(doc_span_index))
                print("tokens: {}".format("".join(tokens_list[i])))
                print("token_to_orig_map: {}".format(token_to_orig_map_list[i]))
                print("token_is_max_context: {}".format(token_is_max_context_list[i]))
                print("input_ids:\n", input_ids_list[i])
                print("start_position:\n", start_position_list[i])
                print("end_position:\n", end_position_list[i])
                print("heads:\n", heads[i])
                print("rels:\n", rels[i])

            if compute_dist:
                features.append({'unique_id': unique_id_list[i],
                                 'example_index': example_index,
                                 'doc_span_index': doc_span_index,
                                 'tokens': tokens_list[i],
                                 'token_to_orig_map': token_to_orig_map_list[i],
                                 'token_is_max_context': token_is_max_context_list[i],
                                 'input_ids': input_ids_list[i],
                                 'attention_mask': attention_mask_list[i],
                                 'token_type_ids': token_type_ids_list[i],
                                 'start_position': start_position_list[i],
                                 'end_position': end_position_list[i],
                                 'heads': heads[i].to_sparse(),
                                 'rels': rels[i].to_sparse(),
                                 'dists': dists[i].to_sparse()})
            else:
                features.append({'unique_id': unique_id_list[i],
                                 'example_index': example_index,
                                 'doc_span_index': doc_span_index,
                                 'tokens': tokens_list[i],
                                 'token_to_orig_map': token_to_orig_map_list[i],
                                 'token_is_max_context': token_is_max_context_list[i],
                                 'input_ids': input_ids_list[i],
                                 'attention_mask': attention_mask_list[i],
                                 'token_type_ids': token_type_ids_list[i],
                                 'start_position': start_position_list[i],
                                 'end_position': end_position_list[i],
                                 'heads': heads[i].to_sparse(),
                                 'rels': rels[i].to_sparse()})
            

    #print('features num:', len(features))
    #json.dump(features, open(output_files[1], 'w'))

    return features


def _convert_index(index, pos, M=None, is_start=True):
    if pos >= len(index):
        pos = len(index) - 1
    if index[pos] is not None:
        return index[pos]
    N = len(index)
    rear = pos
    while rear < N - 1 and index[rear] is None:
        rear += 1
    front = pos
    while front > 0 and index[front] is None:
        front -= 1
    assert index[front] is not None or index[rear] is not None
    if index[front] is None:
        if index[rear] >= 1:
            if is_start:
                return 0
            else:
                return index[rear] - 1
        return index[rear]
    if index[rear] is None:
        if M is not None and index[front] < M - 1:
            if is_start:
                return index[front] + 1
            else:
                return M - 1
        return index[front]
    if is_start:
        if index[rear] > index[front] + 1:
            return index[front] + 1
        else:
            return index[rear]
    else:
        if index[rear] > index[front] + 1:
            return index[rear] - 1
        else:
            return index[front]


def load_and_cache_cmrc2018_examples(args, task, tokenizer, data_type='train', 
                                     return_examples=False, return_features=False):
    cached_examples_file = os.path.join(args.data_dir, 'cached_examples_{}_{}'.format(
            data_type,
            str(task)))

    if args.parser_model is None:
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
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
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_parsed_{}_{}'.format(
            data_type,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task),
            parser_info,
            args.parser_expand_type))

    data_file = os.path.join(args.data_dir, data_type+'.json')
    if os.path.exists(cached_features_file) and os.path.exists(cached_examples_file):
        logger.info("Loading examples from cached file %s", cached_examples_file)
        examples = torch.load(cached_examples_file)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Saving examples to file %s", cached_examples_file)
        examples = read_cmrc2018_examples(data_file, is_training=True if data_type in ['train','dev'] else False)

        torch.save(examples, cached_examples_file)

        logger.info("Creating features from dataset file at %s", args.data_dir)

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
                                                    is_training=True if data_type in ['train','dev'] else False,
                                                    max_seq_length=args.max_seq_length)
        else:
            features = convert_parsed_examples_to_features(examples, 
                                                    tokenizer,
                                                    biaffine_parser, 
                                                    is_training=True if data_type in ['train','dev'] else False,
                                                    max_seq_length=args.max_seq_length, 
                                                    expand_type=args.parser_expand_type,
                                                    align_type=args.parser_align_type,
                                                    return_tensor=args.parser_return_tensor,
                                                    compute_dist=args.parser_compute_dist
                                                    )
    
            del biaffine_parser

        torch.save(features, cached_features_file)


    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    if data_type in ['train','dev']:
        # true label
        all_start_positions = torch.tensor([f['start_position'] for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f['end_position'] for f in features], dtype=torch.long)
    else:
        # this will not be used in predict
        all_start_positions = all_attention_mask
        all_end_positions = all_attention_mask


    if args.parser_model is None:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, 
                                all_start_positions, all_end_positions, all_example_index)
    else:
        all_heads = torch.stack([f['heads'] for f in features])
        all_rels = torch.stack([f['rels'] for f in features])
            
        if args.parser_compute_dist:
            all_dists = torch.stack([f['dists'] for f in features])
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, 
                                all_start_positions, all_end_positions, all_example_index, 
                                all_heads, all_rels, all_dists)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, 
                                all_start_positions, all_end_positions, all_example_index, 
                                all_heads, all_rels)

    outputs = (dataset,)
    if return_features:
        outputs += (features,)
    if return_examples:
        outputs += (examples,)
    return outputs

## output

def write_cmrc2018_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, version_2_with_negative=False, null_score_diff_threshold=0.):
    """Write final predictions to the json file and log-odds of null if needed."""
    print("Writing predictions to: %s" % (output_prediction_file))
    print("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature['example_index']].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(tqdm(all_examples)):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature['unique_id']]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature['tokens']):
                        #print ("### start_index >= len(feature['tokens'])")
                        continue
                    if end_index >= len(feature['tokens']):
                        #print ("### end_index >= len(feature['tokens'])")
                        continue
                    if str(start_index) not in feature['token_to_orig_map'] and \
                            start_index not in feature['token_to_orig_map']:
                        #print ("### start_index not in feature['token_to_orig_map']")
                        continue
                    if str(end_index) not in feature['token_to_orig_map'] and \
                            end_index not in feature['token_to_orig_map']:
                        #print ("### end_index not in feature['token_to_orig_map']")
                        continue
                    # remove str since we use torch,save, the index is int instead of str
                    if not feature['token_is_max_context'].get(start_index, False):
                        #print ("### not feature['token_is_max_context'].get(str(start_index), False)")
                        continue
                    if end_index < start_index:
                        #print ("### end_index < start_index")
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        #print ("### length > max_answer_length")
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature['tokens'][pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature['token_to_orig_map'][pred.start_index]
                orig_doc_end = feature['token_to_orig_map'][pred.end_index]
                orig_tokens = example['doc_tokens'][orig_doc_start:(orig_doc_end + 1)]
                tok_text = "".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = float(probs[i])
            output["start_logit"] = float(entry.start_logit)
            output["end_logit"] = float(entry.end_logit)
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example['qid']] = nbest_json[0]["text"]
            all_nbest_json[example['qid']] = nbest_json
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example['qid']] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example['qid']] = ""
            else:
                all_predictions[example['qid']] = best_non_null_entry.text
            all_nbest_json[example['qid']] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")


def compute_cmrc2018_metrics(original_file, prediction_file):
    ground_truth_file = json.load(open(original_file, 'r'))
    prediction_file = json.load(open(prediction_file, 'r'))
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for instance in ground_truth_file["data"]:
        # context_id   = instance['context_id'].strip()
        # context_text = instance['context_text'].strip()
        for para in instance["paragraphs"]:
            for qas in para['qas']:
                total_count += 1
                query_id = qas['id'].strip()
                query_text = qas['question'].strip()
                answers = [x["text"] for x in qas['answers']]

                if query_id not in prediction_file:
                    print('Unanswered question: {}\n'.format(query_id))
                    skip_count += 1
                    continue

                prediction = str(prediction_file[query_id])
                f1 += calc_f1_score(answers, prediction)
                em += calc_em_score(answers, prediction)

    #f1_score = 100.0 * f1 / total_count
    #em_score = 100.0 * em / total_count
    f1_score = f1 / total_count
    em_score = em / total_count

    avg = (em_score + f1_score) * 0.5
    output_result = OrderedDict()
    output_result['avg'] = avg
    output_result['f1'] = f1_score
    output_result['em'] = em_score
    output_result['total'] = total_count
    output_result['skip'] = skip_count

    return output_result
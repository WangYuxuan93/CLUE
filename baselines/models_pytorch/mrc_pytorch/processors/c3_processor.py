import os
import pickle
import json
import logging
import random
import torch

from tqdm import tqdm
from .processor import DataProcessor, compute_distance
from tools import official_tokenization as tokenization
from torch.utils.data import TensorDataset
from neuronlp2.parser import Parser
from neuronlp2.sdp_parser import SDPParser

n_class = 4

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def c3_collate_fn(batch):
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


class c3Processor(DataProcessor):
    def __init__(self, data_dir):
        self.D = [[], [], []]
        self.data_dir = data_dir

        for sid in range(2):
            data = []
            for subtask in ["d", "m"]:
                with open(self.data_dir + "/c3-" + subtask + "-" + ["train.json", "dev.json"][sid],
                          "r", encoding="utf8") as f:
                    data += json.load(f)
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    # [context, choice 0, choice 1, choice 2, choice 3, answer]
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

        with open(self.data_dir + "/test.json", "r", encoding="utf8") as f:
            data = json.load(f)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    # [context, choice 0, choice 1, choice 2, choice 3, answer]
                    d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                    for k in range(len(data[i][1][j]["choice"])):
                        d += [data[i][1][j]["choice"][k].lower()]
                    for k in range(len(data[i][1][j]["choice"]), 4):
                        d += ['无效答案']  # 有些C3数据选项不足4个，添加[无效答案]能够有效增强模型收敛稳定性
                    # for test set we pick the last choice as answer
                    d += [d[-1]]
                    self.D[2] += [d]


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

        return examples


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
        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            #logger.info("tokens: %s" % " ".join(
            #    [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids_list[i]]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask_list[i]]))
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids_list[i]]))
            logger.info("label: %s (id = %d)" % (example.label, label_id_list[i]))
            logger.info("heads: {}".format(heads[i]))
            logger.info("rels: {}".format(rels[i]))

        if compute_dist:
            features[-1].append(
                InputParsedFeatures(input_ids=input_ids_list[i],
                              attention_mask=attention_mask_list[i],
                              token_type_ids=token_type_ids_list[i],
                              label_id=label_id_list[i],
                              heads=heads[i] if heads.is_sparse else heads[i].to_sparse(),
                              rels=rels[i] if rels.is_sparse else rels[i].to_sparse(),
                              dists=dists[i] if dists.is_sparse else dists[i].to_sparse()))
        else:
            features[-1].append(
                InputParsedFeatures(input_ids=input_ids_list[i],
                              attention_mask=attention_mask_list[i],
                              token_type_ids=token_type_ids_list[i],
                              label_id=label_id_list[i],
                              heads=heads[i] if heads.is_sparse else heads[i].to_sparse(),
                              rels=rels[i] if rels.is_sparse else rels[i].to_sparse()))

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


def load_and_cache_c3_examples(args, task, tokenizer, data_type='train', 
                               return_examples=False, return_features=False):

    processor = c3Processor(args.data_dir)
    label_list = processor.get_labels()

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
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        #features = pickle.load(open(cached_features_file, 'rb'))
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        if data_type == 'train':
            examples = processor.get_train_examples()
        elif data_type == 'dev':
            examples = processor.get_dev_examples()
        else:
            examples = processor.get_test_examples()

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

        torch.save(features, cached_features_file)

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
                if args.parser_compute_dist:
                    dists[-1].append(f[i].dists)
        all_heads = torch.stack([torch.stack(tup) for tup in heads])
        all_rels = torch.stack([torch.stack(tup) for tup in rels])
        
        if args.parser_compute_dist:
            all_dists = torch.stack([torch.stack(tup) for tup in dists])
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids,
                                    all_heads, all_rels, all_dists)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, 
                                    all_heads, all_rels)

    outputs = (dataset,)
    if return_features:
        outputs += (features,)
    return outputs

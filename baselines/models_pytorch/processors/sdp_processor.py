# -*- coding: utf-8 -*-

import logging
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from .utils import DataProcessor
from processors.processor import cached_features_filename
from .mappings.ud_mapping import ud_en_label_mapping
from .mappings.sdpv2_mapping import sdpv2_en_label_mapping

logger = logging.getLogger(__name__)

class InputSDPExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, words, 
                 upos_tags=None, xpos_tags=None,
                 heads=None, rels=None,
                 source_heads=None, source_rels=None):
        self.guid = guid
        self.words = words
        self.upos_tags = upos_tags
        self.xpos_tags = xpos_tags
        self.heads = heads
        self.rels = rels
        self.source_heads = source_heads
        self.source_rels = source_rels


    def show(self):
        logger.info("guid={}".format(self.guid))
        logger.info("words={}".format(self.words))
        logger.info("heads={}".format(self.heads))
        logger.info("rels={}".format(self.rels))
        logger.info("upos_tags={}".format(self.upos_tags))
        logger.info("xpos_tags={}".format(self.xpos_tags))
        logger.info("source_heads={}".format(self.source_heads))
        logger.info("source_rels={}".format(self.source_rels))


class InputSDPFeatures(object):
    def __init__(
        self, 
        input_ids, 
        attention_mask, 
        token_type_ids, 
        word_mask,
        first_ids,
        heads, 
        rels
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.word_mask = word_mask
        self.first_ids = first_ids
        self.heads = heads
        self.rels = rels


class InputParsedSDPFeatures(object):
    def __init__(
        self, 
        input_ids, 
        attention_mask, 
        token_type_ids, 
        word_mask,
        first_ids,
        heads, 
        rels, 
        source_heads,
        source_rels,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.word_mask = word_mask
        self.first_ids = first_ids
        self.heads = heads
        self.rels = rels
        self.source_heads = source_heads
        self.source_rels = source_rels


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    num_items = len(batch[0])
    all_src_heads, all_src_rels = None, None
    if num_items == 7:
        all_input_ids, all_attention_mask, all_token_type_ids, all_first_ids, all_word_mask, all_heads, all_rels = map(torch.stack, zip(*batch))
    elif num_items == 9:
        all_input_ids, all_attention_mask, all_token_type_ids, all_first_ids, all_word_mask, all_heads, all_rels, all_src_heads, all_src_rels = map(torch.stack, zip(*batch))
    max_len = max(all_attention_mask.sum(-1)).item()
    max_word_len = max(all_word_mask.sum(-1)).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_first_ids = all_first_ids[:, :max_word_len]
    all_word_mask = all_word_mask[:, :max_word_len]
    if all_heads.is_sparse:
        all_heads = all_heads.to_dense()
    if all_rels.is_sparse:
        all_rels = all_rels.to_dense()
    all_heads = all_heads[:, :max_word_len, :max_word_len]
    all_rels = all_rels[:, :max_word_len, :max_word_len]
    if num_items == 9:
        if all_src_heads.is_sparse:
            all_src_heads = all_src_heads.to_dense()
            all_src_rels = all_src_rels.to_dense()
        if len(all_src_heads.size()) == 3:
            all_src_heads = all_src_heads[:, :max_len, :max_len]
        elif len(all_src_heads.size()) == 4:
            all_src_heads = all_src_heads[:, :, :max_len, :max_len]
        all_src_rels = all_src_rels[:, :max_len, :max_len]

    batch = {}
    batch["input_ids"] = all_input_ids
    batch["attention_mask"] = all_attention_mask
    batch["token_type_ids"] = all_token_type_ids
    batch["first_ids"] = all_first_ids
    batch["word_mask"] = all_word_mask
    batch["heads"] = all_heads
    batch["rels"] = all_rels
    batch["src_heads"] = all_src_heads
    batch["src_rels"] = all_src_rels
    return batch


class SdpProcessor(DataProcessor):
    def __init__(self, task=None):
        self.task = task
        self.lan = 'en'

    def _read_conll(self, filename):
        sents = []
        with open(filename, 'r') as f:
            data = f.read().strip().split("\n\n")
            for sent in data:
                lines = sent.strip().split("\n")
                sents.append([line.split() for line in lines])
        return sents

    def get_examples(self, data_dir, data_type="train"):
        """See base class."""
        filename = data_type+".conllu"
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, filename)), data_type)


    def get_labels(self):
        self.label_map = sdpv2_en_label_mapping
        return list(self.label_map.keys())


    def get_label_map(self):
        self.label_map = sdpv2_en_label_mapping
        return self.label_map


    def get_source_label_map(self):
        self.source_label_map = ud_en_label_mapping
        return self.source_label_map


    def get_graph(self, heads):
        seq_len = len(heads)
        graph = np.zeros((seq_len+1, seq_len+1), dtype=np.int) # adding root to first place
        rel_graph = [[None]*(seq_len+1) for _ in range(seq_len+1)]
        for i, heads_ in enumerate(heads): # i+1 is current token id
            items = heads_.split("|")
            for item in items:
                head, rel = item.split(":", 1) # only split the left most :
                graph[i+1, int(head)] = 1
                rel_graph[i+1][int(head)] = rel
        return graph, rel_graph


    def _create_examples(self, sents, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, sent) in enumerate(sents):
            sid = "%s-%s" % (set_type, i)
            words = [line[1] for line in sent]
            upos_tags = [line[3] for line in sent]
            xpos_tags = [line[4] for line in sent]
            #heads = ["<ROOT>"]+[int(line[6]) for line in sent]
            #rels = ["<ROOT>"]+[line[7] for line in sent]
            heads, rels = self.get_graph([line[8] for line in sent])
            source_heads, source_rels = self.get_graph([line[9] for line in sent])
            guid = "%s-%s" % (set_type, len(examples))
            examples.append(
                InputSDPExample(guid=guid, words=words, 
                                upos_tags=upos_tags, xpos_tags=xpos_tags,
                                heads=heads, rels=rels,
                                source_heads=source_heads, source_rels=source_rels))

        return examples


def get_word2token_map(word_ids, lengths, debug=False):
    assert len(word_ids) == len(lengths)
    wid2tid_list = []
    for wids, l in zip(word_ids, lengths):
        wid2tid = []
        prev_word = None
        for i, w in enumerate(wids):
            if i >= l: break
            if len(wid2tid) == 0 or w != prev_word:
                wid2tid.append([i])
                prev_word = w
            else:
                wid2tid[-1].append(i)
        wid2tid_list.append(wid2tid)
        if debug:
            print ("wids:\n", wids)
            print ("wid2tid:\n", wid2tid)
    return wid2tid_list

def align_matrix_heads(
        attention_mask,
        word_ids,
        source_heads,
        source_rels,
        max_length=128,
        source_label_map=None,
        expand_type="word",
        debug=False
    ):
        #print ("attention_mask:\n", attention_mask)
        lengths = [sum(mask) for mask in attention_mask]
        #print ("lengths:\n", lengths)
        #print ("word_ids:\n", word_ids)
        wid2tid_list = get_word2token_map(word_ids, lengths)

        src_t_heads_list = []
        src_t_rels_list = []
        for i in range(len(source_heads)):
            if debug:
                print ("word_ids:\n", word_ids[i])
                print ("wid2tid:\n", wid2tid_list[i])
                print ("source_heads:\n", source_heads[i])
                print ("source_rels:\n", source_rels[i])
            
            src_t_heads = torch.zeros(max_length, max_length, dtype=torch.long)
            src_t_rels = torch.zeros(max_length, max_length, dtype=torch.long)
            wid2tid = wid2tid_list[i]
            if "copy" in expand_type:
                src_heads = torch.from_numpy(source_heads[i])
                src_rels = source_rels[i]
                arc_indices = torch.nonzero(src_heads, as_tuple=False).numpy()
                #print ("arc_indices:\n", arc_indices)
                # copy the arc from first char of the head to all chars consisting its children
                for x,y in arc_indices:
                    label = src_rels[x][y]
                    head_id = wid2tid[y][0]
                    child_ids = wid2tid[x]
                    for child_id in child_ids:
                        # ignore out of range arcs
                        if child_id < max_length and head_id < max_length:
                            src_t_heads[child_id][head_id] = 1
                            src_t_rels[child_id][head_id] = source_label_map[label]
            if debug:
                torch.set_printoptions(profile="full")
                print ("src_t_heads:\n", src_t_heads)
                print ("src_t_rels:\n", src_t_rels)

            if "word" in expand_type:
                # add arc with word_label from following chars to the first char of each word
                for tids in wid2tid:
                    if len(tids) > 1:
                        start_id = tids[0]
                        for cid in tids[1:]:
                            src_t_heads[cid][start_id] = 1
                            src_t_rels[cid][start_id] = source_label_map["<WORD>"]
                if debug:
                    print ("src_t_heads (word arc):\n", src_t_heads)
                    print ("src_t_rels (word arc):\n", src_t_rels)
                    #exit()

            src_t_heads_list.append(src_t_heads.to_sparse())
            src_t_rels_list.append(src_t_rels.to_sparse())
            # delete dense tensor to save mem
            del src_t_heads
            del src_t_rels

        heads = torch.stack(src_t_heads_list, dim=0)
        rels = torch.stack(src_t_rels_list, dim=0)

        if debug:
            print ("heads:\n", heads)
            print ("rels:\n", rels)
            exit()

        return heads, rels


def prepare_input(
        attention_mask,
        word_ids,
        heads,
        rels,
        label_map=None,
        debug=False
    ):
        #print ("attention_mask:\n", attention_mask)
        lengths = [sum(mask) for mask in attention_mask]
        wid2tid_list = get_word2token_map(word_ids, lengths)
        word_lens = [len(h) for h in heads]
        max_word_len = max(word_lens)
        word_masks = []
        for word_len in word_lens:
            word_mask = [1 for _ in range(word_len)]
            while len(word_mask) < max_word_len:
                word_mask.append(0)
            word_masks.append(word_mask)

        first_ids_list = []
        for i, wid2tid in enumerate(wid2tid_list):
            first_ids = [tids[0] for tids in wid2tid[:-1]] # rm the last [SEP] token
            assert len(first_ids) == word_lens[i]
            while len(first_ids) < max_word_len:
                first_ids.append(0)
            first_ids_list.append(first_ids)
            
        heads_list = [torch.zeros(max_word_len, max_word_len, dtype=torch.long) for _ in range(len(heads))]
        # set null rel to -100 so it will be filtered out in criterion
        rels_list = [torch.ones(max_word_len, max_word_len, dtype=torch.long)*-100 for _ in range(len(heads))]
        for i in range(len(heads)):
            arc_indices = torch.nonzero(torch.from_numpy(heads[i]), as_tuple=False).numpy()
            for x,y in arc_indices:
                heads_list[i][x][y] = 1
                rels_list[i][x][y] = label_map[rels[i][x][y]]

        if debug:
            print ("lengths:\n", lengths)
            print ("word_ids:\n", word_ids)
            print ("wid2tid_list:\n", wid2tid_list)
            print ("word_masks:\n", word_masks)
            print ("first_ids_list:\n", first_ids_list)
            print ("heads_list:\n", heads_list)
            print ("rels_list:\n", rels_list)

        return word_masks, heads_list, rels_list, first_ids_list


def convert_examples_to_features(
        examples, 
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True
    ):

    processor = SdpProcessor(task)
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
    
    word_mask, heads, rels, first_ids = prepare_input(
                        attention_mask=tokenized_inputs['attention_mask'],
                        word_ids=[tokenized_inputs.word_ids(i) for i in range(len(examples))],
                        heads=[example.heads for example in examples],
                        rels=[example.rels for example in examples],
                        label_map=processor.get_label_map()
                        )

    features = []
    #labels = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        word_ids = tokenized_inputs.word_ids(batch_index=ex_index)
        token_type_ids = tokenized_inputs['token_type_ids'][ex_index]
        attention_mask = tokenized_inputs['attention_mask'][ex_index]

        if ex_index < 5:
            logger.info("*** Example ***")
            #logger.info("guid: %s" % (example.guid))
            #logger.info("sid: %s" % (example.sid))
            example.show()
            logger.info("word_ids: %s" % (" ".join([str(x) for x in word_ids])))
            logger.info("input_ids: %s" % " ".join([str(x) for x in tokenized_inputs['input_ids'][ex_index]]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in tokenized_inputs['attention_mask'][ex_index]]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in tokenized_inputs['token_type_ids'][ex_index]]))
            torch.set_printoptions(profile="full")
            logger.info("word_mask: %s" % " ".join([str(x) for x in word_mask[ex_index]]))
            logger.info("first_ids: %s" % " ".join([str(x) for x in first_ids[ex_index]]))
            print ("\nheads:\n", heads[ex_index])
            print ("\nrels:\n", rels[ex_index])

        features.append(
            InputSDPFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                          attention_mask=tokenized_inputs['attention_mask'][ex_index],
                          token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                          word_mask=word_mask[ex_index],
                          heads=heads[ex_index],
                          rels=rels[ex_index],
                          first_ids=first_ids[ex_index],))
    return features


def convert_parsed_examples_to_features(
        examples, 
        tokenizer,
        parser,
        max_length=512,
        task=None,
        label_list=None,
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

    processor = SdpProcessor(task)
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

    word_mask, heads, rels, first_ids = prepare_input(
                        attention_mask=tokenized_inputs['attention_mask'],
                        word_ids=[tokenized_inputs.word_ids(i) for i in range(len(examples))],
                        heads=[example.heads for example in examples],
                        rels=[example.rels for example in examples],
                        label_map=processor.get_label_map(),
                    )

    if official_syntax_type == "gold":
        src_heads, src_rels = align_matrix_heads(
                        attention_mask=tokenized_inputs['attention_mask'],
                        word_ids=[tokenized_inputs.word_ids(i) for i in range(len(examples))],
                        source_heads=[example.source_heads for example in examples],
                        source_rels=[example.source_rels for example in examples],
                        max_length=max_length,
                        source_label_map=processor.get_source_label_map(),
                        expand_type=expand_type,
                    )
    else:
        src_heads, src_rels = parser.parse_bpes(
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

    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        word_ids = tokenized_inputs.word_ids(batch_index=ex_index)
        token_type_ids = tokenized_inputs['token_type_ids'][ex_index]
        attention_mask = tokenized_inputs['attention_mask'][ex_index]

        if ex_index < 5:
            logger.info("*** Example ***")
            example.show()
            logger.info("word_ids: %s" % (" ".join([str(x) for x in word_ids])))
            logger.info("input_ids: %s" % " ".join([str(x) for x in tokenized_inputs['input_ids'][ex_index]]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in tokenized_inputs['attention_mask'][ex_index]]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in tokenized_inputs['token_type_ids'][ex_index]]))
            torch.set_printoptions(profile="full")
            logger.info("word_mask: %s" % " ".join([str(x) for x in word_mask[ex_index]]))
            logger.info("first_ids: %s" % " ".join([str(x) for x in first_ids[ex_index]]))
            print ("\nheads:\n", heads[ex_index])
            print ("\nrels:\n", rels[ex_index])
            print ("\nsrc_heads:\n", src_heads[ex_index])
            print ("\nsrc_rels:\n", src_rels[ex_index])


        features.append(
            InputParsedSDPFeatures(input_ids=tokenized_inputs['input_ids'][ex_index],
                          attention_mask=tokenized_inputs['attention_mask'][ex_index],
                          token_type_ids=tokenized_inputs['token_type_ids'][ex_index],
                          word_mask=word_mask[ex_index],
                          heads=heads[ex_index],
                          rels=rels[ex_index],
                          first_ids=first_ids[ex_index],
                          source_heads=src_heads[ex_index] if src_heads.is_sparse else src_heads[ex_index].to_sparse(),
                          source_rels=src_rels[ex_index] if src_rels.is_sparse else src_rels[ex_index].to_sparse()))
    return features


def load_and_cache_examples(args, task, tokenizer, data_type='train', return_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = SdpProcessor(task)

    cached_examples_file = os.path.join(args.data_dir, 'cached_examples_{}_{}'.format(data_type, str(task)))
    if return_examples:
        if not os.path.exists(cached_examples_file):
            examples = processor.get_examples(args.data_dir, data_type)
            if args.local_rank in [-1, 0]:
                logger.info("Saving examples into cached file %s", cached_examples_file)
                torch.save(examples, cached_examples_file)
        else:
            examples = torch.load(cached_examples_file)
    # Load data features from cache or dataset file
    if args.official_syntax_type: # gold or pred official syntax
        parser_info = args.official_syntax_type+"-syntax"
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
        cached_features_file = cached_features_filename(args, task, data_type=data_type)

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        examples = processor.get_examples(args.data_dir, data_type)

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
                                            max_length=args.max_seq_length,
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
    all_first_ids = torch.tensor([f.first_ids for f in features], dtype=torch.long)
    all_word_mask = torch.tensor([f.word_mask for f in features], dtype=torch.long)
    all_heads = torch.stack([f.heads for f in features])
    all_rels = torch.stack([f.rels for f in features])

    if args.parser_model is None and not args.official_syntax_type:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_first_ids, 
                                all_word_mask, all_heads, all_rels)
    else:
        all_src_heads = torch.stack([f.source_heads for f in features])
        all_src_rels = torch.stack([f.source_rels for f in features])
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_first_ids, 
                                all_word_mask, all_heads, all_rels,
                                all_src_heads, all_src_rels)

    if return_examples:
        return dataset, examples
    return dataset

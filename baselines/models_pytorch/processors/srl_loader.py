# -*- coding: utf-8 -*-

import logging
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from processors.processor import cached_features_filename
from processors.argument_label_processor import ArgumentLabelProcessor
from processors.argument_label_processor import convert_examples_to_features as arg_convert_examples_to_features
from processors.argument_label_processor import convert_parsed_examples_to_features as arg_convert_parsed_examples_to_features

from processors.predicate_sense_processor import PredicateSenseProcessor
from processors.predicate_sense_processor import convert_examples_to_features as sense_convert_examples_to_features
from processors.predicate_sense_processor import convert_parsed_examples_to_features as sense_convert_parsed_examples_to_features


logger = logging.getLogger(__name__)

converters = {
    'arg': arg_convert_examples_to_features,
    'sense': sense_convert_examples_to_features,
}
parsed_converters = {
    'arg': arg_convert_parsed_examples_to_features,
    'sense': sense_convert_parsed_examples_to_features,
}
processors = {
    'arg': ArgumentLabelProcessor,
    'sense': PredicateSenseProcessor,
}

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


def load_and_cache_examples(args, task, tokenizer, data_type='train', return_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    task_type = task.split('-')[2]
    processor = processors[task_type](task)
    #elif task_type == 'sense':

    output_mode = 'srl'
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
            features = converters[task_type](examples,
                                            tokenizer,
                                            task=args.task_name,
                                            label_list=label_list,
                                            max_length=args.max_seq_length,
                                            output_mode=output_mode,
                                            pad_on_left=bool(args.model_type in ['xlnet']),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )
        else:
            features = parsed_converters[task_type](examples,
                                                    tokenizer,
                                                    biaffine_parser,
                                                    task=args.task_name,
                                                    label_list=label_list,
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
    all_labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    if args.parser_model is None and not args.official_syntax_type:
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
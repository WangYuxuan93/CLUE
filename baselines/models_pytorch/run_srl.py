# -*- coding: utf-8 -*-
# @Author: bo.shi
# @Date:   2019-12-30 19:26:53
# @Last Modified by:   bo.shi
# @Last Modified time: 2019-12-31 19:49:36
""" Finetuning the library models for sequence classification on CLUE (Bert, ERNIE, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, AutoTokenizer, RobertaConfig)

from transformers import AdamW, get_linear_schedule_with_warmup
from metrics.srl_compute_metrics import compute_metrics
from processors.srl_loader import collate_fn, processors, load_and_cache_examples
from tools.common import seed_everything, save_numpy
from tools.common import init_logger, logger
from tools.progressbar import ProgressBar

from neuronlp2.parser import Parser
from neuronlp2.sdp_parser import SDPParser
from models.semsyn_bert import SemSynBertConfig, SemSynBertForArgumentLabel, SemSynBertForPredicateSense
from models.semsyn_roberta import SemSynRobertaConfig, SemSynRobertaForArgumentLabel, SemSynRobertaForPredicateSense
from models.modeling_bert import BertConfig, BertForArgumentLabel, BertForPredicateSense
from models.modeling_bert import BertForWordLevelArgumentLabel, BertForWordLevelPredicateSense
from models.modeling_roberta import RobertaForArgumentLabel, RobertaForPredicateSense
from models.modeling_roberta import RobertaForWordLevelArgumentLabel, RobertaForWordLevelPredicateSense
from io_utils.srl_writer import write_conll09_predicate_sense, write_conll09_argument_label
import shutil
import re
from pathlib import Path


SCONFIG_CLASSES = {
    'bert': SemSynBertConfig,
    'roberta': SemSynRobertaConfig,
}

SMODEL_CLASSES = {
    'bert-arg': SemSynBertForArgumentLabel,
    'roberta-arg': SemSynRobertaForArgumentLabel,
    'bert-sense': SemSynBertForPredicateSense,
    'roberta-sense': SemSynRobertaForPredicateSense,
}

CONFIG_CLASSES = {
    'bert': BertConfig,
    'roberta': RobertaConfig,
}

MODEL_CLASSES = {
    'bert-arg': BertForArgumentLabel,
    'roberta-arg': RobertaForArgumentLabel,
    'bert-sense': BertForPredicateSense,
    'roberta-sense': RobertaForPredicateSense,
}

WORD_LEVEL_MODEL_CLASSES = {
    'bert-arg': BertForWordLevelArgumentLabel,
    'roberta-arg': RobertaForWordLevelArgumentLabel,
    'bert-sense': BertForWordLevelPredicateSense,
    'roberta-sense': RobertaForWordLevelPredicateSense,
}


def _prepare_inputs(inputs, device, use_dist=False, debug=False):

    #print ("inputs:\n", inputs)
    #exit()
    
    for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

    if "first_ids" in inputs and inputs["first_ids"] is None:
        del inputs["first_ids"]
        del inputs["word_mask"]
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

def delete_old_checkpoints(output_dir, best_checkpoint, save_limit=1):
    # delete old checkpoints other than the best
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"checkpoint-*")]

    for path in glob_checkpoints:
        regex_match = re.match(f".*checkpoint-([0-9]+)", path)
        if regex_match and regex_match.groups():
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    
    #print ("checkpoints:\n", checkpoints_sorted)
    if best_checkpoint:
        checkpoints_sorted.remove(best_checkpoint)
        if save_limit == 1:
            checkpoints_to_delete = checkpoints_sorted
        else:
            checkpoints_to_delete = checkpoints_sorted[:-save_limit+1]
    else:
        checkpoints_to_delete = checkpoints_sorted[:-save_limit]
    for checkpoint in checkpoints_to_delete:
        logger.info("Deleting older checkpoint [{}] due to save_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )
    #WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    trainer_state = {'best_metric':0, 'best_checkpoint':None, 'epoch': int(args.num_train_epochs)}
    log_history = []
    steps_every_epoch = len(train_dataloader) / args.train_batch_size
    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            #batch = tuple(t.to(args.device) for t in batch)
            inputs = _prepare_inputs(batch, args.device)
            #print ("batch:\n", batch)
            """
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet', 'albert',
                                                                           'roberta'] else None  # XLM, DistilBERT don't use segment_ids
            """

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                result = None
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print(" ")
                    log_history.append({'epoch': epoch, 'step': global_step, 'loss': loss.item()})
                    # Log metrics
                    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("Epoch = {}, Global step = {}".format(epoch, global_step))
                        result = evaluate(args, model, tokenizer)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if result is not None:
                        _result = {'epoch': epoch, 'step': global_step, 'eval_loss': result['eval_loss']}
                        for key in result:
                            if key.startswith('n_'): continue
                            if not key.startswith('eval'):
                                _result['eval_'+key] = result[key]
                        log_history.append(_result)
                    decisive_metric = 'f1' if args.task_name.endswith('arg') else 'acc'
                    if result is None or result[decisive_metric] > trainer_state['best_metric']:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if result is not None and result[decisive_metric] > trainer_state['best_metric']:
                                trainer_state['best_checkpoint'] = output_dir
                                trainer_state['best_metric'] = result[decisive_metric]
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        tokenizer.save_pretrained(output_dir)
                        #tokenizer.save_vocabulary(vocab_path=output_dir)

                        delete_old_checkpoints(args.output_dir, trainer_state['best_checkpoint'])

        print(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()

    trainer_state['global_step'] = global_step
    trainer_state['log_history'] = log_history
    state_path = os.path.join(args.output_dir, 'trainer_state.json')
    with open(state_path, 'w') as f:
        json.dump(trainer_state, f,indent=4)

    return global_step, tr_loss / global_step, trainer_state['best_checkpoint']


def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, data_type='dev')
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collate_fn)

        # Eval!
        logger.info("********* Running evaluation {} ********".format(prefix))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = []
        out_label_ids = []
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            #batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = _prepare_inputs(batch, args.device)
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            #print ("logits: ", logits.size())
            #print ("labels: ", inputs['labels'].size())
            preds.extend([l for l in logits.detach().cpu().numpy().argmax(-1)])
            out_label_ids.extend([l for l in inputs['labels'].detach().cpu().numpy()])
            pbar(step)
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_loss = eval_loss / nb_eval_steps
        #preds = np.argmax(preds, axis=-1)
        result = compute_metrics(eval_task, preds, out_label_ids)
        result['eval_loss'] = eval_loss
        results.update(result)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("******** Eval results {} ********".format(prefix))
        for key in sorted(result.keys()):
            logger.info(" dev: %s = %s", key, str(result[key]))
    return results


def predict(args, model, tokenizer, label_list, prefix="", is_ood=False):
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.output_dir,)
    label_map = {i: label for i, label in enumerate(label_list)}
    data_type = 'test-ood' if is_ood else 'test' 

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        pred_dataset, pred_examples = load_and_cache_examples(args, pred_task, tokenizer, data_type=data_type, 
                                                              return_examples=True)
        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset) if args.local_rank == -1 else DistributedSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size,
                                     collate_fn=collate_fn)

        logger.info("******** Running prediction {} ({}) ********".format(prefix, 'Out-Of-Domain' if is_ood else 'In-Domain'))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        nb_pred_steps = 0
        preds = []
        out_label_ids = []
        pbar = ProgressBar(n_total=len(pred_dataloader), desc="Predicting")
        for step, batch in enumerate(pred_dataloader):
            model.eval()
            #batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                """
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if (
                            'bert' in args.model_type or 'xlnet' in args.model_type) else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                """
                inputs = _prepare_inputs(batch, args.device)
                outputs = model(**inputs)
                _, logits = outputs[:2]
            nb_pred_steps += 1
            preds.extend([l for l in logits.detach().cpu().numpy().argmax(-1)])
            out_label_ids.extend([l for l in inputs['labels'].detach().cpu().numpy()])
            pbar(step)
        print(' ')

        result = compute_metrics(pred_task, preds, out_label_ids)
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        logger.info("******** Test results {} ********".format(prefix))
        for key in sorted(result.keys()):
            logger.info(" test: %s = %s", key, str(result[key]))

        true_predict_label = [
            [label_map[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(preds, out_label_ids)
        ]

        filename = "test-ood_prediction.txt" if is_ood else "test_prediction.txt"
        output_submit_file = os.path.join(pred_output_dir, prefix, filename)
        # 保存标签结果
        if args.task_name.endswith('sense'):
            write_conll09_predicate_sense(true_predict_label, pred_examples, output_submit_file)
        else:
            write_conll09_argument_label(true_predict_label, pred_examples, output_submit_file)
                
        # 保存中间预测结果
        #output_logits_file = os.path.join(pred_output_dir, prefix, "test_logits")
        #save_numpy(file_path=output_logits_file, data=preds)


def load_labels_from_json(path):
    null_label = "_<PAD>"
    word_label = "_<WORD>"
    with open(path, 'r') as f:
        data = json.load(f)
        parser_label2id = data["instance2index"]
    if null_label not in parser_label2id:
        parser_label2id[null_label] = len(parser_label2id)
    if word_label not in parser_label2id:
        parser_label2id[word_label] = len(parser_label2id)
    return parser_label2id


def main(): 
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default='conll09-zh-arg', type=str, required=True, choices=['conll09-en-sense','conll09-en-arg'
                        ,'conll09-zh-sense','conll09-zh-arg'],
                        help="The name of the task")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--optim", default="AdamW", type=str, choices=["BERTAdam","AdamW"], help="Optimizer")
    ## SBERT parameters
    #parser.add_argument("--use_gold_syntax", action='store_true', help="Whether to use gold syntax tree")
    parser.add_argument("--official_syntax_type", default=None, type=str, choices=[None, "gold", "pred", "diff", "same"], help="Type of the official syntax used")
    parser.add_argument("--is_word_level", action='store_true', help="Whether use label/heads in word level")
    parser.add_argument("--parser_model", default=None, type=str, help="Parser model's path")
    parser.add_argument("--parser_lm_path", default=None, type=str, help="Parser model's pretrained LM path")
    parser.add_argument("--parser_batch", default=32, type=int, help="Batch size for parser")
    parser.add_argument("--parser_type", default="sdp", type=str, choices=["dp","sdp"], help="Type of the parser")
    parser.add_argument("--parser_expand_type", default="copy", type=str, choices=["copy","word","copy-word"], help="Policy to expand parses")
    parser.add_argument("--parser_align_type", default="jieba", type=str, choices=["jieba","pkuseg","rule"], help="Policy to align subwords in parser")
    parser.add_argument("--parser_return_tensor", action='store_true', help="Whether parser should return a tensor")
    parser.add_argument("--parser_compute_dist", action='store_true', help="Whether parser should also compute distance matrix")

    parser.add_argument("--parser_return_graph_mask", action='store_true', help="Whether parser should return a in form of graph mask")
    parser.add_argument("--parser_n_mask", default=3, type=int, help="Max distance for graph mask")
    parser.add_argument("--parser_mask_types", default="parent:child", type=str, choices=["parent","child","parent:child"], help="Graph mask types (split by :)")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    args.parser_mask_types = args.parser_mask_types.split(":")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    #args.output_dir = args.output_dir #+ '{}'.format(args.model_type)
    #if not os.path.exists(args.output_dir):
    #    os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir + '/{}-{}.log'.format(args.model_type, args.task_name))
    logger.info("cuda version: {}".format(torch.version.cuda))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    seed_everything(args.seed)
    # Prepare CLUE task
    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()
    args.task_type = args.task_name.split('-')[2]
    processor = processors[args.task_type](args.task_name)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.parser_model is not None or args.official_syntax_type:
        config_class = SCONFIG_CLASSES[args.model_type]
        model_class = SMODEL_CLASSES[args.model_type+'-'+args.task_type]
        tokenizer_class = AutoTokenizer
    else:
        config_class = CONFIG_CLASSES[args.model_type]
        if args.is_word_level:
            model_class = WORD_LEVEL_MODEL_CLASSES[args.model_type+'-'+args.task_type]
        else:
            model_class = MODEL_CLASSES[args.model_type+'-'+args.task_type]
        tokenizer_class = AutoTokenizer

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                              num_labels=num_labels, finetuning_task=args.task_name)
        if args.official_syntax_type:
            config.graph["num_rel_labels"] = len(processor.get_syntax_label_map())
        elif args.parser_model is not None:
            label_path = os.path.join(args.parser_model, "alphabets/type.json")
            parser_label2id = load_labels_from_json(label_path)
            config.graph["num_rel_labels"] = len(parser_label2id)

        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                    do_lower_case=args.do_lower_case, use_fast=True, add_prefix_space=True)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.to(args.device)
        
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss, best_checkpoint = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        if best_checkpoint is not None:
            logger.info("Loading best model from: %s", best_checkpoint)
            model = model.from_pretrained(best_checkpoint)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case, use_fast=True,)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case, use_fast=True,)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("%s = %s\n" % (key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case, use_fast=True,)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            predict(args, model, tokenizer, label_list, prefix=prefix)
            if args.task_name.split('-')[1] == 'en':
                predict(args, model, tokenizer, label_list, prefix=prefix, is_ood=True)


if __name__ == "__main__":
    main()

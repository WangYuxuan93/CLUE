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
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer,
                          BertForMultipleChoice, BertForQuestionAnswering)

from transformers import AdamW, get_linear_schedule_with_warmup
from tools.pytorch_optimization import BERTAdam

from metrics.mrc_compute_metrics import compute_metrics
from processors import mrc_output_modes as output_modes
from processors import mrc_processors as processors
from processors import collate_fns
from processors import example_loaders
from processors import label_lists
from processors.chid_processor import ChidRawResult, get_final_predictions
from processors.cmrc2018_processor import write_cmrc2018_predictions, compute_cmrc2018_metrics
from processors.drcd_processor import write_drcd_predictions, compute_drcd_metrics
#from processors import load_and_cache_c3_examples as load_and_cache_examples
from tools.common import seed_everything, save_numpy
from tools.common import init_logger, logger
from tools.progressbar import ProgressBar

from neuronlp2.parser import Parser
from neuronlp2.sdp_parser import SDPParser
from models.semsyn_bert import SemSynBertConfig, SemSynBertForMultipleChoice, SemSynBertForQuestionAnswering
import shutil
import re
from pathlib import Path

#ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig,
#                                                                                RobertaConfig)), ())
#MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
#    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
#    'roberta': (BertConfig, BertForSequenceClassification, BertTokenizer),
#    'albert': (BertConfig, AlbertForSequenceClassification, BertTokenizer)
#}

MODEL_CLASSES = {
    'chid': (BertConfig, BertForMultipleChoice, BertTokenizer),
    'c3': (BertConfig, BertForMultipleChoice, BertTokenizer),
    'cmrc2018': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'drcd': (BertConfig, BertForQuestionAnswering, BertTokenizer),
}

SBERT_MODEL = {
    'chid': SemSynBertForMultipleChoice,
    'c3': SemSynBertForMultipleChoice,
    'cmrc2018': SemSynBertForQuestionAnswering,
    'drcd': SemSynBertForQuestionAnswering
}


def _prepare_inputs(inputs, device, use_dist=False, debug=False):

    #print ("inputs:\n", inputs)
    
    for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

    #print ("start_positions:\n", inputs['start_positions'])
    #print ("end_positions:\n", inputs['end_positions'])

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
                                  collate_fn=collate_fns[args.task_name])

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
    logger.info("  Optimizer: %s", args.optim)
    if args.optim == 'AdamW': 
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )
    elif args.optim == 'BERTAdam':
        optimizer = BERTAdam(params=optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             max_grad_norm=args.max_grad_norm,
                             t_total=t_total,
                             schedule='warmup_linear',
                             weight_decay_rate=args.weight_decay)

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

    if args.output_mode == 'classification':
        args.decision_metric = 'acc'
    trainer_state = {'best_metric':0, 'decision_metric':args.decision_metric, 'best_checkpoint':None, 
                     'epoch': int(args.num_train_epochs)}
    log_history = []
    steps_every_epoch = len(train_dataloader) / args.train_batch_size
    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            #batch = tuple(t.to(args.device) for t in batch)
            inputs = _prepare_inputs(batch, args.device)
            if args.task_name in ['chid','cmrc2018','drcd'] and 'example_indices' in inputs:
                del inputs['example_indices']
            #print ("batch:\n", batch)

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
                if args.optim == 'AdamW':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                if args.optim == 'AdamW':
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                result = None
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print(" ")
                    log_history.append({'step': global_step, 'loss': loss.item()})
                    # Log metrics
                    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("Epoch = {}, Global step = {}".format(epoch, global_step))
                        result = evaluate(args, model, tokenizer, global_step=global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if result is not None:
                        #log_history.append({'step': global_step, 'eval_loss': result['eval_loss'], 
                        #                    'eval_acc': result['acc']})
                        result['step'] = global_step
                        log_history.append(result)
                        if args.output_mode == 'classification':
                            metric = result['acc']
                        else:
                            metric = result[args.decision_metric]

                    if result is None or metric > trainer_state['best_metric']:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if result is not None and metric > trainer_state['best_metric']:
                                trainer_state['best_checkpoint'] = output_dir
                                trainer_state['best_metric'] = metric
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


def evaluate(args, model, tokenizer, prefix="", global_step=0):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if eval_task in ['cmrc2018','drcd']:
            RawResult = collections.namedtuple("RawResult",
                               ["unique_id", "start_logits", "end_logits"])
        if args.output_mode == 'qa':
            eval_dataset, eval_features, eval_examples = example_loaders[args.task_name](args, eval_task, 
                                                            tokenizer, data_type='dev', 
                                                            return_features=True, return_examples=True)
        else:
            eval_dataset, eval_features = example_loaders[args.task_name](args, eval_task, 
                                                            tokenizer, data_type='dev', 
                                                            return_features=True, return_examples=False)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collate_fns[args.task_name])

        if args.task_name == 'chid':
            all_tags = [f.tag for f in eval_features]
            all_example_ids = [f.example_id for f in eval_features]

        # Eval!
        logger.info("********* Running evaluation {} ********".format(prefix))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        all_predictions = []
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            #batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = _prepare_inputs(batch, args.device)
                if args.task_name in ['chid', 'cmrc2018', 'drcd']:
                    example_indices = inputs['example_indices']
                    del inputs['example_indices']

                outputs = model(**inputs)
                if args.output_mode == 'qa':
                    tmp_eval_loss, batch_start_logits, batch_end_logits = outputs
                else:
                    tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if args.output_mode != 'qa':
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            pbar(step)
            
            if args.task_name == 'chid':
                for i, example_index in enumerate(example_indices):
                    logits_ = logits[i].detach().cpu().tolist()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    all_predictions.append(ChidRawResult(unique_id=unique_id,
                                                 example_id=all_example_ids[unique_id],
                                                 tag=all_tags[unique_id],
                                                 logit=logits_))
            elif args.task_name in ['cmrc2018','drcd']:
                #print ("start_logits:\n", batch_start_logits)
                #print ("end_logits:\n", batch_end_logits)
                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature['unique_id'])
                    all_predictions.append(RawResult(unique_id=unique_id,
                                                 start_logits=start_logits,
                                                 end_logits=end_logits))
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_loss = eval_loss / nb_eval_steps
        if args.task_name == 'cmrc2018':
            outpath = os.path.join(args.output_dir, 'tmp_prediction')
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            output_prediction_file = os.path.join(outpath,
                                          "predictions_steps" + str(global_step) + ".json")
            output_nbest_file = output_prediction_file.replace('predictions', 'nbest')
            write_cmrc2018_predictions(eval_examples, eval_features, all_predictions,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)
            dev_file = os.path.join(args.data_dir, 'dev.json')
            result = compute_cmrc2018_metrics(dev_file, output_prediction_file)
        elif args.task_name == 'drcd':
            outpath = os.path.join(args.output_dir, 'tmp_prediction')
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            output_prediction_file = os.path.join(outpath,
                                          "predictions_steps" + str(global_step) + ".json")
            output_nbest_file = output_prediction_file.replace('predictions', 'nbest')
            write_drcd_predictions(eval_examples, eval_features, all_predictions,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)
            dev_file = os.path.join(args.data_dir, 'dev.json')
            result = compute_drcd_metrics(dev_file, output_prediction_file)
        else:
            if args.task_name == 'chid':
                preds = get_final_predictions(all_predictions, g=True)
                #print ("preds:\n", preds)
                preds = [p[1] for p in preds]
            elif args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif args.output_mode == "regression":
                preds = np.squeeze(preds)
            result = compute_metrics(eval_task, preds, out_label_ids)
        result['eval_loss'] = eval_loss
        results.update(result)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("******** Eval results {} ********".format(prefix))
        for key in sorted(result.keys()):
            logger.info(" dev: %s = %s", key, str(result[key]))
    return results


def predict(args, model, tokenizer, label_list, prefix=""):
    if args.task_name in ['cmrc2018','drcd']:
        RawResult = collections.namedtuple("RawResult",
                               ["unique_id", "start_logits", "end_logits"])
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.output_dir,)
    label_map = {i: label for i, label in enumerate(label_list)}

    print ("task:", args.task_name)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        if args.output_mode == 'qa':
            pred_dataset, pred_features, pred_examples = example_loaders[args.task_name](args, pred_task, 
                                                            tokenizer, data_type='test', 
                                                            return_features=True, return_examples=True)
        else:
            pred_dataset, pred_features = example_loaders[args.task_name](args, pred_task, 
                                                            tokenizer, data_type='test', 
                                                            return_features=True, return_examples=False)
        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset) if args.local_rank == -1 else DistributedSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size,
                                     collate_fn=collate_fns[args.task_name])

        if args.task_name == 'chid':
            all_tags = [f.tag for f in pred_features]
            all_example_ids = [f.example_id for f in pred_features]

        logger.info("******** Running prediction {} ********".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        nb_pred_steps = 0
        preds = None
        pbar = ProgressBar(n_total=len(pred_dataloader), desc="Predicting")
        all_predictions = []
        for step, batch in enumerate(pred_dataloader):
            model.eval()
            #batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = _prepare_inputs(batch, args.device)
                if args.task_name in ['chid', 'cmrc2018', 'drcd']:
                    example_indices = inputs['example_indices']
                    if 'example_indices' in inputs:
                        del inputs['example_indices']
                    if 'labels' in inputs:
                        del inputs['labels']
                    if 'start_positions' in inputs:
                        del inputs['start_positions']
                    if 'end_positions' in inputs:
                        del inputs['end_positions']

                outputs = model(**inputs)
                if args.output_mode == 'qa':
                    batch_start_logits, batch_end_logits = outputs
                elif len(outputs) == 1:
                    logits = outputs[0]
                else:
                    _, logits = outputs[:2]
            nb_pred_steps += 1

            if args.output_mode != 'qa':
                if preds is None:
                    if pred_task == 'copa':
                        preds = logits.softmax(-1).detach().cpu().numpy()
                    else:
                        preds = logits.detach().cpu().numpy()
                else:
                    if pred_task == 'copa':
                        preds = np.append(preds, logits.softmax(-1).detach().cpu().numpy(), axis=0)
                    else:
                        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            pbar(step)

            if args.task_name == 'chid':
                for i, example_index in enumerate(example_indices):
                    logits_ = logits[i].detach().cpu().tolist()
                    pred_feature = pred_features[example_index.item()]
                    unique_id = int(pred_feature.unique_id)
                    all_predictions.append(ChidRawResult(unique_id=unique_id,
                                                 example_id=all_example_ids[unique_id],
                                                 tag=all_tags[unique_id],
                                                 logit=logits_))
            elif args.task_name in ['cmrc2018','drcd']:
                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    pred_feature = pred_features[example_index.item()]
                    unique_id = int(pred_feature['unique_id'])
                    all_predictions.append(RawResult(unique_id=unique_id,
                                                 start_logits=start_logits,
                                                 end_logits=end_logits))

        print(' ')
        if args.task_name == 'cmrc2018':
            output_submit_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
            output_nbest_file = output_submit_file.replace('prediction', 'nbest')
            write_cmrc2018_predictions(pred_examples, pred_features, all_predictions,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_submit_file,
                      output_nbest_file=output_nbest_file)
        elif args.task_name == 'drcd':
            output_submit_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
            output_nbest_file = output_submit_file.replace('prediction', 'nbest')
            write_drcd_predictions(pred_examples, pred_features, all_predictions,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_submit_file,
                      output_nbest_file=output_nbest_file)
        else:
            if args.task_name == 'chid':
                preds = get_final_predictions(all_predictions, g=True)
                #print ("preds:\n", preds)
            elif args.output_mode == "classification":
                predict_label = np.argmax(preds, axis=1)
            elif args.output_mode == "regression":
                predict_label = np.squeeze(preds)
            if pred_task == 'copa':
                predict_label = []
                pred_logits = preds[:, 1]
                i = 0
                while (i < len(pred_logits) - 1):
                    if pred_logits[i] >= pred_logits[i + 1]:
                        predict_label.append(0)
                    else:
                        predict_label.append(1)
                    i += 2
            output_submit_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
            output_logits_file = os.path.join(pred_output_dir, prefix, "test_logits")
            # 保存标签结果
            with open(output_submit_file, "w") as writer:
                if args.task_name == 'chid':
                    json_d = {}
                    for tag, pred in preds:
                        json_d[tag] = pred
                    json.dump(json_d, writer, indent=4)
                else:
                    for i, pred in enumerate(predict_label):
                        json_d = {}
                        json_d['id'] = i
                        json_d['label'] = str(label_map[pred])
                        writer.write(json.dumps(json_d) + '\n')
            # 保存中间预测结果
            save_numpy(file_path=output_logits_file, data=preds)


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
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## SBERT parameters
    parser.add_argument("--parser_model", default=None, type=str, help="Parser model's path")
    parser.add_argument("--parser_lm_path", default=None, type=str, help="Parser model's pretrained LM path")
    parser.add_argument("--parser_batch", default=32, type=int, help="Batch size for parser")
    parser.add_argument("--parser_type", default="sdp", type=str, choices=["dp","sdp"], help="Type of the parser")
    parser.add_argument("--parser_expand_type", default="copy", type=str, choices=["copy","word","copy-word"], help="Policy to expand parses")
    parser.add_argument("--parser_align_type", default="jieba", type=str, choices=["jieba","pkuseg","rule"], help="Policy to align subwords in parser")
    parser.add_argument("--parser_return_tensor", action='store_true', help="Whether parser should return a tensor")
    parser.add_argument("--parser_compute_dist", action='store_true', help="Whether parser should also compute distance matrix")
    parser.add_argument("--parser_use_reverse_label", action='store_true', help="Whether use reversed parser label")

    ## QA parameters
    parser.add_argument('--decision_metric', type=str, default='em', choices=['em','f1','avg'], help="use which metric to chose QA model")
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=20)

    ## Other parameters
    parser.add_argument("--optim", default="AdamW", type=str, choices=["AdamW","BERTAdam"], help="Type of optimizer")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_num_choices", default=10, type=int,
                        help="The maximum number of cadicate answer,  shorter than this will be padded.")
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

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    #args.output_dir = args.output_dir #+ '{}'.format(args.model_type)
    #if not os.path.exists(args.output_dir):
    #    os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir + '/{}-{}.log'.format(args.model_type, args.task_name))
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
    #if args.task_name not in processors:
    #    raise ValueError("Task not found: %s" % (args.task_name))
    #processor = processors[args.task_name](args.data_dir)
    args.output_mode = output_modes[args.task_name]
    #label_list = processor.get_labels()
    #num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.parser_model is not None:
        config = SemSynBertConfig.from_pretrained(
                        args.config_name if args.config_name else args.model_name_or_path)
        #config.num_labels=num_labels

        tokenizer = BertTokenizer.from_pretrained(
                        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
        config.use_reverse_rel = args.parser_use_reverse_label
        label_path = os.path.join(args.parser_model, "alphabets/type.json")
        parser_label2id = load_labels_from_json(label_path)
        config.graph["num_rel_labels"] = len(parser_label2id)
        
        model_class = SBERT_MODEL[args.task_name]
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        tokenizer_class = BertTokenizer
    else:
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.task_name]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                              finetuning_task=args.task_name)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                    do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = example_loaders[args.task_name](args, args.task_name, tokenizer, data_type='train')[0]
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
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
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
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
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
            predict(args, model, tokenizer, label_lists[args.task_name], prefix=prefix)


if __name__ == "__main__":
    main()

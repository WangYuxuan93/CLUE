#!/usr/bin/env bash
# @Author: bo.shi
# @Date:   2019-11-04 09:56:36
# @Last Modified by:   bo.shi
# @Last Modified time: 2020-01-01 11:46:07

TASK_NAME="ocnli"
MODEL_NAME="bert-base-chinese"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
#export BERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
export BERT_PRETRAINED_MODELS_DIR=/mnt/hgfs/share/chinese-roberta-wwm-ext
export BERT_WWM_DIR=$BERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
export GLUE_DATA_DIR=$CURRENT_DIR/CLUEdatasets
PARSER_MODEL=/mnt/hgfs/share/parser_model/zh-news-biaf-basic

# make output dir
if [ ! -d $CURRENT_DIR/${TASK_NAME}_output ]; then
  mkdir -p $CURRENT_DIR/${TASK_NAME}_output
  echo "makedir $CURRENT_DIR/${TASK_NAME}_output"
fi

# run task
cd $CURRENT_DIR
echo "Start running..."
if [ $# == 0 ]; then
    python run_classifier.py \
      --model_type=bert \
      --model_name_or_path=$BERT_PRETRAINED_MODELS_DIR \
      --task_name=$TASK_NAME \
      --parser_type=sdp \
      --parser_model=$PARSER_MODEL \
      --config_name=config/semsyn_bert.json \
      --parser_lm_path=$BERT_PRETRAINED_MODELS_DIR \
      --parser_return_tensor \
      --parser_align_type pkuseg \
      --parser_expand_type copy \
      --do_train \
      --do_eval \
      --do_predict \
      --do_lower_case \
      --data_dir=$GLUE_DATA_DIR/${TASK_NAME}/ \
      --max_seq_length=16 \
      --per_gpu_train_batch_size=16 \
      --per_gpu_eval_batch_size=16 \
      --learning_rate=2e-5 \
      --num_train_epochs=2.0 \
      --max_steps=0 \
      --logging_steps=10 \
      --save_steps=10 \
      --output_dir=$CURRENT_DIR/${TASK_NAME}_output/model \
      --overwrite_output_dir \
      --seed=42
elif [ $1 == "predict" ]; then
    echo "Start predict..."
    python run_classifier.py \
      --model_type=bert \
      --model_name_or_path=$BERT_PRETRAINED_MODELS_DIR \
      --task_name=$TASK_NAME \
      --parser_model=$PARSER_MODEL \
      --parser_lm_path=$BERT_PRETRAINED_MODELS_DIR \
      --parser_compute_dist \
      --parser_return_tensor \
      --do_predict \
      --do_lower_case \
      --data_dir=$GLUE_DATA_DIR/${TASK_NAME}/ \
      --max_seq_length=16 \
      --per_gpu_train_batch_size=16 \
      --per_gpu_eval_batch_size=16 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --logging_steps=3335 \
      --save_steps=3335 \
      --output_dir=$CURRENT_DIR/${TASK_NAME}_output/model \
      --overwrite_output_dir \
      --seed=42
fi

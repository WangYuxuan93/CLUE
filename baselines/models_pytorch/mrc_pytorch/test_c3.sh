#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=roberta_wwm_ext_base
export OUTPUT_DIR=$CURRENT_DIR/check_points
export BERT_PRETRAINED_MODELS_DIR=/mnt/hgfs/share/chinese-roberta-wwm-ext
export GLUE_DIR=$CURRENT_DIR/mrc_data # set your data dir
PARSER_MODEL=/mnt/hgfs/share/parser_model/zh-news-biaf-basic
TASK_NAME="c3"

python run_c3.py \
  --model_name_or_path=$BERT_PRETRAINED_MODELS_DIR \
  --task_name=$TASK_NAME \
  --config_name=data/sbertv2.json \
  --parser_type=sdp \
  --parser_model=$PARSER_MODEL \
  --parser_lm_path=$BERT_PRETRAINED_MODELS_DIR \
  --parser_return_tensor \
  --do_train \
  --do_eval \
  --data_dir=$GLUE_DIR/$TASK_NAME/ \
  --max_seq_length=16 \
  --per_gpu_train_batch_size=2 \
  --per_gpu_eval_batch_size=2 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --gradient_accumulation_steps=1 \
  --max_steps=-1 \
  --logging_steps=3 \
  --save_steps=3 \
  --warmup_proportion=0.05 \
  --output_dir=$OUTPUT_DIR/$TASK_NAME/ \
  --overwrite_output_dir \
  --seed 42

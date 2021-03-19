#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=roberta_wwm_ext_base
export OUTPUT_DIR=$CURRENT_DIR/check_points
export BERT_DIR=/mnt/hgfs/share/chinese-roberta-wwm-ext
export GLUE_DIR=$CURRENT_DIR/mrc_data # set your data dir
TASK_NAME="cmrc2018"

python run_mrc.py \
  --train_epochs=2 \
  --save_steps=10 \
  --logging_steps=10 \
  --n_batch=16 \
  --gradient_accumulation_steps=4 \
  --lr=3e-5 \
  --warmup_rate=0.1 \
  --max_seq_length=32 \
  --task_name=$TASK_NAME \
  --decision_metric=em \
  --model_name_or_path=$BERT_DIR \
  --train_dir=$GLUE_DIR/$TASK_NAME/train_features.json \
  --train_file=$GLUE_DIR/$TASK_NAME/train.json \
  --dev_dir1=$GLUE_DIR/$TASK_NAME/dev_examples.json \
  --dev_dir2=$GLUE_DIR/$TASK_NAME/dev_features.json \
  --dev_file=$GLUE_DIR/$TASK_NAME/dev.json \
  --output_dir=$CURRENT_DIR/${TASK_NAME}_output/model \

exit

python test_mrc.py \
  --gpu_ids="0" \
  --n_batch=32 \
  --max_seq_length=512 \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_restore_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --test_dir1=$GLUE_DIR/$TASK_NAME/test_examples.json \
  --test_dir2=$GLUE_DIR/$TASK_NAME/test_features.json \
  --test_file=$GLUE_DIR/$TASK_NAME/cmrc2018_test_2k.json \





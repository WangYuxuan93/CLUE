#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=roberta_wwm_ext_base
export OUTPUT_DIR=$CURRENT_DIR/check_points
export BERT_DIR=/mnt/hgfs/share/chinese-roberta-wwm-ext
export GLUE_DIR=$CURRENT_DIR/mrc_data # set your data dir
TASK_NAME="chid"

python run_multichoice_mrc.py \
  --num_train_epochs=3 \
  --save_steps=1 \
  --logging_steps=1 \
  --train_batch_size=16 \
  --predict_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --learning_rate=2e-5 \
  --warmup_proportion=0.06 \
  --max_seq_length=32 \
  --model_name_or_path=$BERT_DIR \
  --input_dir=$GLUE_DIR/$TASK_NAME/ \
  --output_dir=$CURRENT_DIR/${TASK_NAME}_output/model \
  --train_file=$GLUE_DIR/$TASK_NAME/train.json \
  --train_ans_file=$GLUE_DIR/$TASK_NAME/train_answer.json \
  --predict_file=$GLUE_DIR/$TASK_NAME/dev.json \
  --predict_ans_file=$GLUE_DIR/$TASK_NAME/dev_answer.json

exit

python test_multichoice_mrc.py \
  --gpu_ids="0" \
  --predict_batch_size=24 \
  --max_seq_length=64 \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/config.json \
  --input_dir=$GLUE_DIR/$TASK_NAME/ \
  --init_restore_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --predict_file=$GLUE_DIR/$TASK_NAME/test.json \

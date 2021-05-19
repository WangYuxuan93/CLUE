#!/usr/bin/env bash
# @Author: bo.shi
# @Date:   2019-11-04 09:56:36
# @Last Modified by:   bo.shi
# @Last Modified time: 2020-01-01 11:46:07

lan=zh
TASK_NAME="conll09-$lan-srl"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export BERT_PRETRAINED_MODELS_DIR=/mnt/hgfs/share/models/$lan-roberta-base
#export BERT_PRETRAINED_MODELS_DIR=/mnt/hgfs/share/models/roberta-base
export DATA_DIR=$CURRENT_DIR/data/$lan
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
    python run_end2end_srl.py \
      --model_type=bert \
      --model_name_or_path=$BERT_PRETRAINED_MODELS_DIR \
      --task_name=$TASK_NAME \
      --is_word_level \
      --parser_type=sdp \
      --config_name=config/semsyn_$lan.json \
      --parser_lm_path=$BERT_PRETRAINED_MODELS_DIR \
      --parser_align_type pkuseg \
      --parser_expand_type copy-word \
      --parser_n_mask 3 \
      --parser_mask_types parent:child \
      --do_predict \
      --do_lower_case \
      --data_dir=$DATA_DIR/ \
      --max_seq_length=32 \
      --per_gpu_train_batch_size=16 \
      --per_gpu_eval_batch_size=2 \
      --learning_rate=5e-5 \
      --num_train_epochs=30.0 \
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

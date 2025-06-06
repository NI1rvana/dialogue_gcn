#!/bin/bash

dataset=iemocap
dataset_dir=data/${dataset}/dataset
ckpt_dir=data/${dataset}/ckpt
do_what=$1

if [ "${do_what}" == "preprocess" ]; then
  python -u preprocess.py --data=${ckpt_dir}/data.pkl \
      --dataset=${dataset} > log/preprocess.${dataset}
  sleep 100
elif [ "${do_what}" == "train" ]; then
  python -u train.py --data=${ckpt_dir}/data.pkl \
      --from_begin --device=cuda:7 --epochs=50 --drop_rate=0.4 \
      --weight_decay=0.0 --class_weight --batch_size=8 --learning_rate=0.0003 > log/train.${dataset}
      sleep 100
fi
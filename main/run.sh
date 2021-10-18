#!/bin/bash

if [ "$1" = "script-test" ]; then
  python script_test.py

elif [ "$1" = "pretrain" ]; then
  python run.py -task-type-run="pretrain"
elif [ "$1" = "meta-train" ]; then
  python run.py -task-type-run="meta-train"
elif [ "$1" = "meta-train-with-pretrain" ]; then
  python run.py -task-type-run="meta-train-with-pretrain" -path-params="../result/pretrain_BPD_ALL_RT/model/CNN, epoch[19], ACC[0.371].pt"
elif [ "$1" = "meta-finetune" ]; then
  python run.py -task-type-run="meta-finetune" -path-params="../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt"
elif [ "$1" = "meta-test" ]; then
  python run.py -task-type-run="meta-test" -path-params='../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt' -path-config='../result/pretrain_meta_train_BPD_ALL_RT_MIMML/config.pkl'
elif [ "$1" = "meta-inference" ]; then
  python run.py -task-type-run="meta-inference" -path-params='../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt' -path-config='../result/pretrain_meta_train_BPD_ALL_RT_MIMML/config.pkl'

else
  echo "Invalid Option Selected"
fi

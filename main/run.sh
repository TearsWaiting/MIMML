#!/bin/bash

if [ "$1" = "script_test" ]; then
	python script_test.py
elif [ "$1" = "pretrain_BPD45" ]; then
  python run.py pretrain
else
	echo "Invalid Option Selected"
fi
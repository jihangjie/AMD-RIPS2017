#!/bin/bash

DATA_DIR="data/svhn"

DATASETS=("train_32x32.mat" "test_32x32.mat")

for DATA in "${DATASETS[@]}"; do
  if ! [ -e $DATA_DIR$DATA ]; then
    echo downloading $DATA...
    wget -P $DATA_DIR "http://ufldl.stanford.edu/housenumbers/"$DATA
  else
    echo already have $DATA...
  fi
done

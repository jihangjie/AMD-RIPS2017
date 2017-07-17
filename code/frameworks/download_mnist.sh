#!/bin/bash

DATA_DIR="data/mnist/"

DATASETS=("train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz" \
          "t10k-images-idx3-ubyte.gz" "t10k-labels-idx1-ubyte.gz")

for DATA in "${DATASETS[@]}"; do
  if ! [ -e $DATA_DIR$DATA ]; then
    echo downloading $DATA...
    wget -P $DATA_DIR "http://yann.lecun.com/exdb/mnist/"$DATA
    gunzip $DATA_DIR$DATA
  else
    echo already have $DATA...
  fi
done

echo done!

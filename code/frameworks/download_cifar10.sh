#!/bin/bash

DATA_DIR="data/cifar10/"

DATASET="cifar-10-python.tar.gz"
URL="http://www.cs.toronto.edu/~kriz/"

if ! [ -e $DATA_DIR$DATASET ]; then
  echo downloading $DATA...
  wget -P $DATA_DIR $URL$DATASET
else
  echo already have $DATASET...
fi

tar xzvf $DATA_DIR$DATASET && mv cifar-10-batches-py $DATA_DIR

echo done!

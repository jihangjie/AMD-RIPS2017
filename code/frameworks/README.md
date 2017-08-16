### Running the framework

Note: Anything you need to run will be in the parent directory `framework`,
not in any sub-directories.

In order to run the CNN, you must first:
* have Theano and NumPy packages installed
* download the dataset you wish to use (e.g. run `download_mnist.sh` for MNIST), 
one can download the dataset online and place it under data/<dataset_name> 
directory (e.g. the MNIST dataset could be put under data/mnist)

You can then run the batch/layer python scripts. Feel free to edit the
parameters as necessary for what you would like to test. By default, the scripts 
runs batch-by-batch truncation.  

In `by_batch`, you can find our model for batch-by-batch truncation.
In `by_layer`, you can find our modelfor layer-by-layer truncation.
In `helper_files`, you can find our truncation/stochastic rounding script,
as well as data-loading scripts.

We also have a more complicated network under `dropout`, which uses a dropout
layer, momentum method, Adam optimizer, 2 conv layers & 1 dense layer, and
stochastic rounding.

### Plotting Code

There are a few versions of plotting code in this directory:
* `plot.py`
* `truncate_plot.py`
* `convmist_plot.py`

`plot.py` plots any data between 0 and 1 against the sequence of numbers from
1 to `len(data)` and saves them ubder the subdirectory `plots/`.
You can specify which filenames to plot through command-line arguments, by
running `python plot.py <filenames>`. If no filenames are specified, then
`plot.py` will look for data in the subdirectory `data/`.

`convmist_plot.py` was a specific version of `plot.py` that plotted accuracy
over iterations. It is used the same way that `plot.py` is, but shows the data
differently. Differences between `plot.py` and `convmist_plot.py` include:
* name under which files are saved
* x-axis scaled differently (for this exp. we printed accuracy once every 10
  batches instead of after each batch to increase runtime)
* location of legend
* axes labels and title
Note: this script was used to plot
`AMD_RIPS2017/code/plots/Convolutional MNIST: Different Precisions.png`.

`truncate_plot.py` was another specific version of `plot.py` that plotted
accuracy vs. bitsize, rather than iterations. It is used the same way that
`plot.py` is, but shows the data differently. Differences between `plot.py` and
`truncate_plot.py` include:
* name under which files are saved
* location and labels of legend
* axes labels and title
Note: this script was used to plot
`AMD_RIPS2017/code/plots/Different Precisions.png`.

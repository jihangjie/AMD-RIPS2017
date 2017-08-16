# convmist_plot.py
# plots accuracy data (or any set of numbers between 0 and 1) against sequence
# of 1 to len(data), saves them under subdirectory 'plots/'
# if no filenames provided by command-line args, will look for data in
# the subdirectory 'data/'
# plots data from regular_convmist.py

import matplotlib.pyplot as plt
from collections import OrderedDict
import glob, os, sys

def parse_data(filename):
  ''' parse data separated by newlines
  @param filename name of file containing data
  @return list of data as floats
  '''
  # check file existence
  if not os.path.isfile(filename):
    print("File {} does not exist! Exiting".format(filename))
    exit(1)
  with open(filename, 'r') as f:
    data = f.readlines()
  try:
    data = [float(line.strip()) for line in data]
  except:
    print("Issue with {}: expecting numeric values separated by newlines. Exiting".format(filename))
    exit(1)
  return data

def plot_data(textfiles):
  ''' plot data
  @param filename name of file containing data
  @param data list of data as floats
  '''
  # remove ext from filename for plot title/saving

  NUM_COLORS=10
  cm = plt.get_cmap('gist_rainbow')
  fig = plt.figure()
  ax= fig.add_subplot(111)
  ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
  for filename in sorted(textfiles):
    data = parse_data(filename)
    filename = filename.rsplit('.', 1)[0]
    filename = "".join(filename.split('data/'))
    savename = ""
    label = filename.split("_")[1]
    print "label is {}".format(label)

    # label plot (axes, title, etc)
    x = [num for num in range(1, len(data)+1)]
    ax.plot(x, data, label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(savename, y=1.05)
    plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size':9})
    plt.axis([1, 50, 0, 1])
    plt.subplots_adjust(right=0.8)
  print("plotted data from {}".format(filename))

def main():
  ''' if textfiles given as arguments, plot data from those files
  else plot data from all files in 'data/'
  note: all plots will be placed in subdirectory "plots/"
  '''
  args = sys.argv[1:]
  if len(args) is 0:
    textfiles = glob.glob('data/trunclayer/2C1D/*.txt')
  else:
    textfiles = []
    while len(args):
      textfiles.append(args.pop(0))
  if len(textfiles) is 0:
    print("No data to plot! Exiting")
    exit(0)

  # check for existence of "plots" dir
  try:
    os.stat('./plots')
  except:
    os.mkdir('./plots')

  plot_data(textfiles)
  plt.savefig("plots/trunclayer/Truncation by Layer: Two Convolution Layers & One Dense Layer")
  
if __name__ == "__main__":
  main()

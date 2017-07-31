# convmist_plot.py
# plots accuracy data (or any set of numbers between 0 and 1) against sequence
# of 1 to len(data), saves them under subdirectory 'plots/'
# if no filenames provided by command-line args, will look for data in
# the subdirectory 'data/'
# plots data from regular_convmist.py

import matplotlib.pyplot as plt
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

def plot_data(filename, data):
  ''' plot data
  @param filename name of file containing data
  @param data list of data as floats
  '''
  # remove ext from filename for plot title/saving
  filename = filename.rsplit('.', 1)[0]
  filename = "".join(filename.split('data/'))
  savename = "MNIST Convolutional: Accuracy over Time"
  label = filename.split("_")[2]
  print "label is {}".format(label)

  # label plot (axes, title, etc)
  x = [num*10 for num in range(1, len(data)+1)]
  plt.plot(x, data, label=label)
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.title(savename)

  plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size':9})
  plt.axis([10, 500, 0, 1])
  plt.subplots_adjust(right=0.8)

def main():
  ''' if textfiles given as arguments, plot data from those files
  else plot data from all files in 'data/'
  note: all plots will be placed in subdirectory "plots/"
  '''
  args = sys.argv[1:]
  if len(args) is 0:
    textfiles = glob.glob('data/*.txt')
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
  
  for name in sorted(textfiles):
    data = parse_data(name)
    plot_data(name, data)
    print("plotted data from {}".format(name))
  plt.savefig("plots/Convolutional MNIST: Different Precisions")
  
if __name__ == "__main__":
  main()

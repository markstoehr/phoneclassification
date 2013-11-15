from __future__ import division
import numpy as np
import argparse

def main(args):
    labels = np.load(args.labels)
    print "\n".join(tuple( str(k) for k in np.sort(np.unique(labels))))

if __name__=="__main__":
    parser = argparse.ArgumentParser(""" Simple script to get all the unique entries in a vector to be printed to standard output after being sorted""")
    parser.add_argument('--labels',
                        type=str,
                        help='path to the file that you want to get the sorted, unique entries from')
    main(parser.parse_args())

from __future__ import division
import numpy as np
import argparse

def main(args):
    labels = np.load(args.labels)
    components = np.load(args.components)
    print "\n".join(tuple( str(k) for k in np.sort(np.unique(components[labels==args.label]))))

if __name__=="__main__":
    parser = argparse.ArgumentParser(""" Simple script to get all the unique entries in a vector to be printed to standard output after being sorted""")
    parser.add_argument('--label',
                        type=int,
                        help='label that you are concerned with')
    parser.add_argument('--labels',
                        type=str,
                        help='path to the file that you want to get the sorted, unique entries from')

    parser.add_argument('--components',
                        type=str,
                        help='path to the components file that you want to get the sorted, unique entries from that match the given label')
    main(parser.parse_args())

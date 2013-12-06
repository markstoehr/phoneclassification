import numpy as np
from src.hog_gray_scale import hog
import argparse

def main(args):
    """
    Read in a data file and compute the HOG features
    """
    phns2ids = dict(tuple( l.split() for l in open(args.phns2ids,'r').read().strip().split('\n')))

    # read in the files
    input_labels_fls = dict(tuple( l.split() for l in open(args.infile,'r').read().strip().split('\n')))

    # go through the labels
    labels = np.sort(input_labels_fls.keys())

    fhandle = open(args.output_fl_list,'w')
    # open the files and save to a new location
    for label in labels:
        print label
        if args.ftype=='ascii':
            X = np.loadtxt(input_labels_fls[label])
            T = X.shape[1]-args.nnuisance_dimensions
            Y = X[:,-args.nnuisance_dimensions:] # get the nuisance dimensions out
            X = X[:,:T].reshape(X.shape[0],T/args.feature_stride,
                                args.feature_stride)


            H0 = hog(X[0].T,sbin=5)
            H = np.zeros((len(X),
                         H0.size + args.nnuisance_dimensions))
            for i, x in enumerate(X):
                H[i,:H0.size] = hog(x.T,sbin=5).ravel()
                H[i,H0.size:] = Y[i,:]

            fname = '%s_%s.npy' % (args.output_prefix,phns2ids[label])

            np.save( fname,H)
            fhandle.write('%s %s\n' % (label, fname))

    fhandle.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("""
    compute hog features for all the data files given in a file list
    """)
    parser.add_argument("--infile",type=str,
                        help='path to the list of files containing the data to be converted')
    parser.add_argument('--feature_stride',type=int,
                        default=40,help='if this is not none the data is assumed to be a matrix of 1d vectors where each vector corresponds to an instance and has a vector stride for the fastest axis (assumed to be features)')
    parser.add_argument('--nnuisance_dimensions',type=int,
                        default=0,help='if this is set then the number of nuisance dimensions to ignore in the feature dimension and added on to the HOG vector')
    parser.add_argument('--ftype',type=str,
                        help='type of data file to expect, determines how the data is loaded')
    parser.add_argument('--output_prefix',type=str,
                        help='prefix to put in front of the saved output files')
    parser.add_argument('--output_fl_list',type=str,
                        help='prefix to put in front of the saved output files')
    parser.add_argument('--phns2ids',type=str,
                        help='file containing two columns where the first is the list of phones and the second is the index corresponding to that phone label')
    main(parser.parse_args())

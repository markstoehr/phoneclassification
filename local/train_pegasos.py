from __future__ import division
import numpy as np
import argparse, itertools
from src import pegasos

def main(args):
    """
    get all the data matrices and process the data
    """
    phones = np.loadtxt(args.phones,dtype=str)


    avgs = np.load('%s/avgs_%s' % (args.in_prefix, args.in_suffix))
    counts = np.load('%s/counts_%s' % (args.in_prefix, args.in_suffix))

    avgs1=avgs[args.phone1]
    avgs2=avgs[args.phone2]
    counts1=counts[args.phone1]
    counts2=counts[args.phone2]

    avgs1 = np.clip(
        avgs1,
        1/(2*counts1),
        1 - 1/(2*counts1))
    avgs2 = np.clip(
        avgs2,
        1/(2*counts2),
        1 - 1/(2*counts2))

    w_init = np.zeros(avgs1.size+1,dtype=float)
    w_init[:-1] = (np.log( avgs1 * (1-avgs2)/( avgs2 * (1-avgs1)))).ravel()
    w_init[-1] = np.log( 1-avgs1).sum() - np.log(1-avgs2).sum()

    # get the number of phones for the first phone
    X1_shape = np.load('%s/%s_%s' % ( args.data_prefix,
                                   phones[args.phone1],
                                   args.data_suffix)).shape
    N1 = X1_shape[0]
    dim = np.prod(X1_shape[1:])
    N2 = len(np.load('%s/%s_%s' % ( args.data_prefix,
                                    phones[args.phone2],
                                   args.data_suffix)))
    
    X = np.ones((N1+N2,dim+1),dtype=np.int8)
    X[:N1,:-1] = np.load('%s/%s_%s' % ( args.data_prefix,
                                    phones[args.phone1],
                                   args.data_suffix)).reshape(N1,dim)
    X[N1:,:-1] = np.load('%s/%s_%s' % ( args.data_prefix,
                                    phones[args.phone2],
                                   args.data_suffix)).reshape(N2,dim)
    y = np.zeros(N1+N2,dtype=int)
    y[:N1] = 1
    y[N1:] = -1

    X *= y[:,np.newaxis]

    n_correct = (np.dot(X,w_init) > 0).sum()
    print "Percent correct before training: %g" % (n_correct/len(X))
    w = pegasos.perform_batch_updates(X,
                                      args.T,
                                      args.l,
                                      args.k,w_init=w_init,
                                      v=args.v)

    n_correct = (np.dot(X,w[-1]) > 0).sum()

    print "Percent correct on training: %g" % (n_correct/len(X))

    np.save(args.out_w,w[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("""
    calculate the models for every phone and output them
    along with a vector indicating the phone they belong to
    """)
    parser.add_argument('--phones',type=str,help='path to list of phones')
    parser.add_argument('--phone1',type=int,help='integer indicating the index of the first phone to use')
    parser.add_argument('--phone2',type=int,help='integer indicating the index of the second phone to use')
    parser.add_argument('--T',type=int,help='integer indicating the number of iterations to run pegasos')
    parser.add_argument('--k',type=int,help='integer indicating the size of the mini-batch to use for pegasos')
    parser.add_argument('--l',type=float,help='float indicating the regularization term for pegasos with a larger term corresponding to greater regularization')
    parser.add_argument('--data_prefix',type=str,
                        help='prefix of files that contain the data')
    parser.add_argument('--data_suffix',type=str,
                        help='suffix for the data files')

    parser.add_argument('--in_prefix',type=str,
                        help='prefix of files that contain the avgs and counts')
    parser.add_argument('--in_suffix',type=str,
                        help='suffix for the avgs and counts files')
    parser.add_argument('--out_w',type=str,
                        help='path to save the output classifier to')
    parser.add_argument('-v',action='store_true',help='use verbose output')
    main(parser.parse_args())
    



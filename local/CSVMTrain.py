#!/usr/bin/python

from __future__ import division
import numpy as np
import argparse, itertools
import matplotlib.pyplot as plt
from sklearn import svm
from template_speech_rec import configParserWrapper
from scipy.spatial.distance import pdist, squareform, cdist

def main(args):
    """
    """
    config_d = configParserWrapper.load_settings(open(args.config,'r'))


    y = []
    for fpath_id,fpath in enumerate(args.input_data_matrices):
        X0 = np.load(fpath)
        X0_shape = X0.shape[1:]
        y.extend(fpath_id * np.ones(len(X0)))

    y = np.array(y)

    # add in two coordinates for the length
    if args.input_lengths is not None:
        X = np.zeros((len(y),np.prod(X0_shape)+2),dtype=float)
    else:
        X = np.zeros((len(y),np.prod(X0_shape)),dtype=float)
    cur_idx = 0
    for fpath_id,fpath in enumerate(args.input_data_matrices):
        X0 = np.load(fpath)

        X[cur_idx:cur_idx+len(X0),:np.prod(X0_shape)] = X0.reshape(len(X0),
                                                np.prod(X0_shape))
        cur_idx += len(X0)





    if args.input_lengths is not None:
        train_ls = []
        for fpath in args.input_lengths:
            ls = np.loadtxt(fpath,dtype=int)
            train_ls.extend(ls[:,2])

        train_ls = np.log(np.tile(np.array(train_ls),
                           (2,1)).T)
        train_ls[:,1] *= train_ls[:,1]
        X[:,-2:] = train_ls





    X_test = []
    y_test = []
    for fpath_id,fpath in enumerate(args.input_test_data_matrices):
        X0 = np.load(fpath).astype(np.float)
        X0_shape = X0.shape[1:]
        X_test.extend(X0.reshape(len(X0),
                            np.prod(X0_shape)))
        y_test.extend(fpath_id * np.ones(len(X0)))

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if args.input_test_lengths is not None:
        test_ls = []
        for fpath in args.input_test_lengths:
            ls = np.loadtxt(fpath,dtype=int)
            test_ls.extend(ls[:,2])

        test_ls = np.log(np.tile(np.array(test_ls),
                           (2,1)).T)
        test_ls[:,1] *= test_ls[:,1]

        X_test = np.hstack((X_test,test_ls))


    if args.use_weights is not None:
        weights = np.load(args.use_weights)
        weights = weights.reshape(weights.size)
        X *= weights
        X_test *= weights
        if args.do_kernel_dist_features:
            
            X_test = np.exp(-cdist(X_test,X, 'euclidean'))
            X = np.exp(-squareform(pdist(X, 'euclidean')))


    penalty_names = config_d['SVM']['penalty_list'][::2]
    penalty_values = tuple( float(k) for k in config_d['SVM']['penalty_list'][1::2])

    dev_results = ()
    exp_descriptions = ()
    exp_description_id = 0

    for penalty_name, penalty_value in itertools.izip(penalty_names,penalty_values):
        if args.v:
            print '%s %s' % ('linear', penalty_name)

        if config_d['SVM']['kernel'] == 'linear':
            clf = svm.LinearSVC(C=penalty_value,
                                loss='l1')
            clf.fit(X,y)
        elif config_d['SVM']['kernel'] == 'polynomial':

            import pdb; pdb.set_trace()

        np.save('%s_%s_%s_coef.npy' % (args.output_fls_prefix,config_d['SVM']['kernel'],
                                           penalty_name),
                    clf.coef_[0])
        np.save('%s_%s_%s_intercept.npy' % (args.output_fls_prefix,config_d['SVM']['kernel'],
                                           penalty_name),
                    clf.intercept_)

        y_test_hat = clf.predict(X_test)
        exp_descriptions += ((config_d['SVM']['kernel'],penalty_name),)

        dev_results += ( (exp_description_id,
                          np.abs(y_test_hat-y_test).sum()/len(y_test), # error rate
                          np.abs(y_test_hat[y_test==0]-y_test[y_test==0]).sum()/len(y_test[y_test==0]), # mistakes by class 0
                          np.abs(y_test_hat[y_test==1]-y_test[y_test==1]).sum()/len(y_test[y_test==1]) # mistakes by class 1
                          ),)
        if args.v:
            print '\t'.join(tuple( str(k) for k in dev_results[-1]))
        exp_description_id +=1

    open('%s_exp_descriptions' % args.output_fls_prefix,
         'w').write('\n'.join(tuple(
             '%d %s' % (k,
                        ' '.join(d))
             for k,d in enumerate(exp_descriptions))))
    np.save('%s_dev_results.npy' % args.output_fls_prefix,
            np.array(dev_results))





if __name__=="__main__":
    parser = argparse.ArgumentParser("""
    Train the svm and save the statistics about the
    filters to a directory.  This is a wrapper for the linear
    SVM provided by scikit learn.
    """)
    parser.add_argument('--input_data_matrices',
                        type=str,
                        nargs='+',
                        help='list of paths to input matrices, generally assumed to only accept two')
    parser.add_argument('--input_test_data_matrices',
                        type=str,
                        nargs='+',
                        help='list of paths to input matrices, generally assumed to only accept two should be in the same order as in the --input_data_matrices arguments')
    parser.add_argument('--input_lengths',
                        type=str,
                        nargs='+',
                        default=None,
                        help='list of paths to the lengths of the input examples of training data, Default is None in which case nothing is included')
    parser.add_argument('--input_test_lengths',
                        type=str,
                        nargs='+',
                        default=None,
                        help='list of paths to the lengths of the input examples of testing data, Default is None in which case nothing is included')
    parser.add_argument('--output_fls_prefix',
                        type=str,
                        help='prefix for where all the different output files are going to be saved')
    parser.add_argument('--config',
                        type=str,
                        default='conf/main.config',
                        help='configuration file')
    parser.add_argument('--train_config',
                        type=str,
                        default='conf/train.config',
                        help='train config that will be read and written to')
    parser.add_argument('--use_weights',
                        type=str,
                        default=None,
                        help='weights for the feature vector')
    parser.add_argument('--do_kernel_dist_features',
                        action='store_true',
                        help='whether to use the Gaussian kernel to get the SVM features, generally for use with with a weight vector')
    parser.add_argument('-v',
                        action='store_true',
                        help='verbosity flag')
    main(parser.parse_args())

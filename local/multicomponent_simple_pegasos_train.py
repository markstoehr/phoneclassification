from __future__ import division
from src import bernoullimm
import numpy as np
import argparse
from src.pegasos import multiclass, multiclass_regularize_diffs

def main(args):
    rootdir = args.rootdir
    confdir='%s/%s'% (rootdir,args.confdir)
    datadir='%s/%s' % (rootdir,args.datadir)
    expdir='%s/%s' % (rootdir,args.expdir)


    leehon=np.loadtxt('%s/%s' % (confdir,
                                 args.leehon_phones),dtype=str)
    phones39 = np.unique(np.sort(leehon[:,1]))
    phones39_dict = dict( (v,i) for i,v in enumerate(phones39))
    phones48_dict = dict( (v,i) for i,v in enumerate(leehon[:,0]))
    leehon_dict = dict( (phones48_dict[p],
                     phones39_dict[q]) for p,q in leehon)


    nobs = np.zeros(len(leehon[:,0]))
    for phone_id, phone in enumerate(leehon[:,0]):
        X = np.load('%s/%s_%s' % (datadir,phone,args.train_data_suffix))
        nobs[phone_id] = X.shape[0]
        dim = np.prod(X.shape[1:])

    X = np.ones((nobs.sum(),dim+1),dtype=np.uint8)
    y = np.zeros(nobs.sum(),dtype=int)
    y39 = np.zeros(nobs.sum(),dtype=int)
    curobs=0
    for phone_id, phone in enumerate(leehon[:,0]):
        X[curobs:curobs+nobs[phone_id],:-1] = np.load('%s/%s_%s' % (datadir,phone,args.train_data_suffix)).reshape(nobs[phone_id],dim)
        y[curobs:curobs+nobs[phone_id]] = phone_id
        y39[curobs:curobs+nobs[phone_id]] = leehon_dict[phone_id]
        curobs += nobs[phone_id]

    # do random weight initialization
    T = int(args.nrounds_multiplier * X.shape[0] )
    print T
    avgs_meta_list = np.loadtxt('%s/%s' % (expdir,args.avgs_meta_list_fl),
                                dtype=str)
    for identifier, avg_fl, meta_fl, l, do_projection, regularize_diffs in avgs_meta_list:
        print identifier, avg_fl, meta_fl, l, do_projection, regularize_diffs
        if do_projection == 'False':
            do_projection = False
        else:
            do_projection = 50
        if regularize_diffs == 'False':
            regularize_diffs = False
        else:
            regularize_diffs = True


        l = float(l)
        avgs = np.load('%s/%s' % (expdir,avg_fl))
        avgs = avgs.reshape(avgs.shape[0],
                    dim)
        W_init = np.zeros((len(avgs),dim+1))
        W_init[:,:-1] = np.log(avgs) - np.log(1-avgs)
        W_init[:,-1] = np.log(1-avgs).sum(-1)

        W_meta = np.load('%s/%s' % (expdir,meta_fl))
        if regularize_diffs:
            print "using multiclass_regularize_diffs"
            W = W_init.copy()
            for cur_batch in xrange(int(T/int(X.shape[0]/2))):
                np.save('%s/%s_w_iter%d_%dniter_%s%s' % (expdir,
                                  args.save_prefix,cur_batch,
                                                         T,
                                  identifier,
                                  args.save_suffix),
                W)
                W =multiclass_regularize_diffs(y,X,int(X.shape[0]/2),l,1,W.copy(),W_meta,v=False,loss='hinge',
                                           do_projection=do_projection)
            np.save('%s/%s_w_iter%d_%dniter_%s%s' % (expdir,
                                  args.save_prefix,cur_batch+1,T,
                                  identifier,
                                  args.save_suffix),
                W)
            W =multiclass_regularize_diffs(y,X,(T % int(X.shape[0]/2)),l,1,W.copy(),W_meta,v=False,loss='hinge',
                                           do_projection=do_projection)

        else:
            print "using multiclass"
            W = W_init.copy()
            start_t = 500
            for cur_batch in xrange(int(T/int(X.shape[0]/4))):
                np.save('%s/%s_w_iter%d_%dniter_%s%s' % (expdir,
                                  args.save_prefix,cur_batch,
                                                         T,
                                  identifier,
                                  args.save_suffix),
                W)
                W =multiclass(y,X,int(X.shape[0]/4),l,1,W.copy(),W_meta,start_t=start_t,v=False,loss='hinge',
                                           do_projection=do_projection)
                start_t += int(X.shape[0]/4)
            np.save('%s/%s_w_iter%d_%dniter_%s%s' % (expdir,
                                  args.save_prefix,cur_batch+1,T,
                                  identifier,
                                  args.save_suffix),
                W)
            W =multiclass(y,X,(T % int(X.shape[0]/4)),l,1,W.copy(),W_meta,start_t=start_t,v=False,loss='hinge',
                                           do_projection=do_projection)


        np.save('%s/%s_w_%dniter_%s%s' % (expdir,
                                  args.save_prefix,
                                          T,
                                  identifier,
                                  args.save_suffix),
                W)
        
        


if __name__=="__main__":
    parser = argparse.ArgumentParser("""Train pegasos using a warm start mixture model""")
    parser.add_argument('--rootdir',
                        default='/home/mark/Research/phoneclassification',help='root dir for the phone classification')
    parser.add_argument('--confdir',
                        default='conf',
                        help='location where configuration files are kept')
    parser.add_argument('--datadir',
                        default='data/local/data',
                        help='where the data is located')
    parser.add_argument('--expdir',
                        default='exp/pegasos',
                        help='where to save the outputs from the experiment files')
    parser.add_argument('--leehon_phones',
                        default='phones.48-39',
                        help='file containing the 48 to 39 conversion')
    parser.add_argument('--train_data_suffix',
                        default='train_examples.npy',
                        help='the suffix for the training data files minus a leading underscore')
    parser.add_argument('--nrounds_multiplier',type=float,
                        default=2,
                        help='number of times to run through the dataset')
    parser.add_argument('--avgs_meta_list_fl',type=str,
                        default='avgs_meta_list_fl',
                        help='six column file where first column is the identifier we want used in saving the file, second column is for where the avgs are saved, third column is where the meta file is saved, fourth column is where the lambda parameter is kept, the fifth column is whether or not to do the projection, and the sixth column indicates whether we are using the regularized differences version of the algorithm. The assumption is that the file paths for saved files are relative to the experiment directory `args.exp`')
    parser.add_argument('--save_prefix',
                        default='warm_train_pegasos',
                        help='')
    parser.add_argument('--save_suffix',
                        default='.npy',
                        help='')
    main(parser.parse_args())


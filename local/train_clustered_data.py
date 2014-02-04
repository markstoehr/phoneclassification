from __future__ import division
from phoneclassification.confusion_matrix import confusion_matrix
import numpy as np
import argparse,collections
from phoneclassification.pegasos import multiclass_multicomponent_polyavg

"""
Extract the set of data associated with a set of phones
and give a label set, also initialize the components using the
basic components from the model
"""

def get_reduced_meta(W_meta,leehon_dict):
    W_meta_leehon = W_meta.copy()
    # keep a record of the component id under the
    # reduced set of labels
    component_count = collections.defaultdict(int)
    for w_id, w in enumerate(W_meta):
        W_meta_leehon[w_id,0] = leehon_dict[w[0]]
        W_meta_leehon[w_id,1] = component_count[W_meta_leehon[w_id,0]]
        component_count[W_meta_leehon[w_id,0]] += 1

    return W_meta_leehon

parser = argparse.ArgumentParser("""File to run a basic test of the pegasos multiclass
SVM solver over the scattering features""")
parser.add_argument('--root_dir',default='/home/mark/Research/phoneclassification',type=str,help='root directory for where to look for things')
parser.add_argument('--data_dir',default='data/local/data',type=str,
                    help='relative path to where the data is kept')

parser.add_argument('--cluster_features',type=str,help='path to where the features will be saved')
parser.add_argument('--cluster_labels',type=str,help='path to where the labels for the cluster saved-- -1 is reserved for the background labels')
parser.add_argument('--cluster_models',type=str,help='path to where the models will be saved')
parser.add_argument('--cluster_meta',type=str,help='path to where the model metadata will be saved')
parser.add_argument('--save_prefix',type=str,help='prefix for path to save the output to')
parser.add_argument('-l',type=float,nargs='+',help='lambda scaling parameter to be using')
parser.add_argument('--eta',type=float,nargs='+',help='different values of eta to try for the polynomial averaging')
parser.add_argument('-T',type=int,nargs='+',help='Number of iterations to run this for')
# parser.add_argument('--',type=,help='')
args = parser.parse_args()

rootdir = args.root_dir[:]
confdir='%s/conf'%rootdir
datadir='%s/%s' % (rootdir,args.data_dir)


leehon=np.loadtxt('%s/phones.48-39' % confdir,dtype=str)
phones39 = np.unique(np.sort(leehon[:,1]))
phones39_dict = dict( (v,i) for i,v in enumerate(phones39))
phones48_dict = dict( (v,i) for i,v in enumerate(leehon[:,0]))
leehon_dict = dict( (phones48_dict[p],
                     phones39_dict[q]) for p,q in leehon)


# handle background
leehon_dict[-1] = -1

X = np.load(args.cluster_features)
y = np.load(args.cluster_labels)
y39 = np.array(tuple(leehon_dict[k] for k in y))

W = np.load(args.cluster_models)
W_meta = np.load(args.cluster_meta).astype(int)

# need to construct W_meta39 to use 39 labels
W_meta39 = get_reduced_meta(W_meta,leehon_dict).astype(int)

for eta in args.eta:
    print "Using eta = %g" % eta
    for l in args.l:
        print "Using lambda = %g " % l
        poly_avg_W = W.copy()
        W_trained = W.copy()
        start_t = 1
        for T in args.T:
            print "number of iterations T= %d" % (T+start_t)
            W_trained, poly_avg_W, W_loss, poly_avg_W_loss = multiclass_multicomponent_polyavg(y39,X,T,l,W_trained,W_meta39,eta,start_t=start_t,loss_computation=2000,
               return_avg_W=True,return_loss=True,verbose=True,loss='hinge',
               do_projection=False)

            start_t += T
            np.save('%s_%gl_%dT_W.npy' % (args.save_prefix,l,start_t-1), W_trained)
            np.save('%s_%gl_%dT_W_loss.npy' % (args.save_prefix,l,start_t-1), W_loss)
            np.save('%s_polyavg_%gl_%geta_%dT_W.npy' % (args.save_prefix,l,eta,start_t-1), poly_avg_W)
            np.save('%s_polyavg_%gl_%geta_%dT_W_loss.npy' % (args.save_prefix,l,eta,start_t-1), poly_avg_W_loss)



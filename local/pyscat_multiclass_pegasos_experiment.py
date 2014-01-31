from __future__ import division
import numpy as np
import argparse
from phoneclassification.pegasos import multiclass


parser = argparse.ArgumentParser("""File to run a basic test of the pegasos multiclass
SVM solver over the scattering features""")
parser.add_argument('--root_dir',default='/home/mark/phoneclassification',type=str,help='root directory for where to look for things')
parser.add_argument('--data_dir',default='data/local/data',type=str,
                    help='relative path to where the data is kept')
parser.add_argument('--exp_dir',default='exp/multiclass_pegasos',type=str,help='experiment directory to save the outputs to')
args = parser.parse_args()

rootdir = args.root_dir[:]
confdir='%s/conf'%rootdir
datadir='%s/%s' % (rootdir,args.data_dir)
expdir=args.exp_dir[:]


leehon=np.loadtxt('%s/phones.48-39' % confdir,dtype=str)
phones39 = np.unique(np.sort(leehon[:,1]))
phones39_dict = dict( (v,i) for i,v in enumerate(phones39))
phones48_dict = dict( (v,i) for i,v in enumerate(leehon[:,0]))
leehon_dict = dict( (phones48_dict[p],
                     phones39_dict[q]) for p,q in leehon)


nobs = np.zeros(len(leehon[:,0]))
for phone_id, phone in enumerate(leehon[:,0]):
    X = np.loadtxt('%s/msc_features_13t_40f_%s_0.dat' % (datadir,phone))
    nobs[phone_id] = X.shape[0]
    dim = np.prod(X.shape[1:])
    print dim

X = np.ones((nobs.sum(),dim+1),dtype=np.uint8)
y = np.zeros(nobs.sum(),dtype=int)
y39 = np.zeros(nobs.sum(),dtype=int)
curobs=0
for phone_id, phone in enumerate(leehon[:,0]):
    X[curobs:curobs+nobs[phone_id],:-1] = np.loadtxt('%s/msc_features_13t_40f_%s_0.dat' % (datadir,phone)).reshape(nobs[phone_id],dim)
    y[curobs:curobs+nobs[phone_id]] = phone_id
    y39[curobs:curobs+nobs[phone_id]] = leehon_dict[phone_id]
    curobs += nobs[phone_id]


for l in (3.0)**(np.arange(2,-5,-1)):
    print "l=%g" % l
    W = np.zeros((48,X.shape[1]))
    for T in [100,1000,10000,100000,1000000,1e7]:
        print "T=%d" % T
        for start_t in [100,1000,10000,100000]:
            print "start_t=%d" % start_t
            W, W_avg, W_loss, W_avg_loss = multiclass(y,X,T,l,W,start_t=1,loss_computation=10000,
                                                      return_avg_W=True,return_loss=True,verbose=True,loss='hinge',
               do_projection=False)
            np.save('%s/multiclass_1ex_pegasos_pyscat_W_noproj_%gl_%dT_%dStartT.npy' % (expdir, l,T,start_t), W)
            np.save('%s/multiclass_1ex_pegasos_pyscat_W_avg_noproj_%gl_%dT_%dStartT.npy' % (expdir, l,T,start_t), W_avg)
            np.save('%s/multiclass_1ex_pegasos_pyscat_W_loss_noproj_%gl_%dT_%dStartT.npy' % (expdir, l,T,start_t), W_loss)
            np.save('%s/multiclass_1ex_pegasos_pyscat_W_avg_loss_noproj_%gl_%dT_%dStartT.npy' % (expdir, l,T,start_t), W_avg_loss)

# load in the data and create a data matrix to be saved

from __future__ import division
from phoneclassification.confusion_matrix import confusion_matrix
import numpy as np

rootdir='/home/mark/Research/phoneclassification'
confdir='%s/conf'%rootdir
datadir='%s/data/local/data' % rootdir
expdir='%s/exp/pegasos' % rootdir

leehon=np.loadtxt('%s/phones.48-39' % confdir,dtype=str)
phones39 = np.unique(np.sort(leehon[:,1]))
phones39_dict = dict( (v,i) for i,v in enumerate(phones39))
phones48_dict = dict( (v,i) for i,v in enumerate(leehon[:,0]))
leehon_dict = dict( (phones48_dict[p],
                     phones39_dict[q]) for p,q in leehon)


nobs = np.zeros(len(leehon[:,0]))
for phone_id, phone in enumerate(leehon[:,0]):
    X = np.load('%s/%s_dev_examples.npy' % (datadir,phone))
    nobs[phone_id] = X.shape[0]
    dim = np.prod(X.shape[1:])
    print dim

X = np.ones((nobs.sum(),dim+1),dtype=np.uint8)
y = np.zeros(nobs.sum(),dtype=int)
y39 = np.zeros(nobs.sum(),dtype=int)
curobs=0
for phone_id, phone in enumerate(leehon[:,0]):
    X[curobs:curobs+nobs[phone_id],:-1] = np.load('%s/%s_dev_examples.npy' % (datadir,phone)).reshape(nobs[phone_id],dim)
    y[curobs:curobs+nobs[phone_id]] = phone_id
    y39[curobs:curobs+nobs[phone_id]] = leehon_dict[phone_id]
    curobs += nobs[phone_id]



for exp_dir in ['pairwise_bernoulli_thresh', 'pegasos']:
    if exp_dir == 'pairwise_bernoulli_thresh':
        n_init = 5
    else:
        n_init = 10
    for nmix in [2,3,6,9,12]:
        avgs = np.load('exp/%s/avgs_%dC.npy' % (exp_dir,nmix))
        avgs = avgs.reshape(avgs.shape[0],
                    dim)
        W_init = np.zeros((len(avgs),dim+1))
        W_init[:,:-1] = np.log(avgs) - np.log(1-avgs)
        W_init[:,-1] = np.log(1-avgs).sum(-1)
        W_meta = np.load('exp/%s/meta_%dC.npy' % (exp_dir,nmix))
        scores = np.dot(X,W_init.T)
        yhat39 = np.array(tuple( leehon_dict[k] for k in  W_meta[:,0][scores.argmax(1)] ))
        print exp_dir, nmix, np.sum(yhat39 == y39)/len(y39)
        open('exp/%s/leehon39_bmm_dev_%dNinit_%dC_error_rate.txt' % (exp_dir,n_init,nmix),'w').write(str(np.sum(yhat39 == y39)/len(y39)))

for exp_dir in ['pegasos']:
    if exp_dir == 'pairwise_bernoulli_thresh':
        n_init = 5
    else:
        n_init = 10
    for nmix in [2,3,6,15,18,21,24,27, 40,50,60,70]:
        avgs = np.load('exp/%s/avgs_%dC.npy' % (exp_dir,nmix))
        avgs = avgs.reshape(avgs.shape[0],
                    dim)
        W_init = np.zeros((len(avgs),dim+1))
        W_init[:,:-1] = np.log(avgs) - np.log(1-avgs)
        W_init[:,-1] = np.log(1-avgs).sum(-1)
        W_meta = np.load('exp/%s/meta_%dC.npy' % (exp_dir,nmix))
        scores = np.dot(X,W_init.T)
        yhat39 = np.array(tuple( leehon_dict[k] for k in  W_meta[:,0][scores.argmax(1)] ))
        C = confusion_matrix(yhat39,y39)
        np.save('exp/%s/leehon39_bmm_dev_%dNinit_%dC_confusion_matrix.npy' % (exp_dir,n_init,nmix),C)
        print exp_dir, nmix, np.sum(yhat39 == y39)/len(y39)
        open('exp/%s/leehon39_bmm_dev_%dNinit_%dC_error_rate.txt' % (exp_dir,n_init,nmix),'w').write(str(np.sum(yhat39 == y39)/len(y39)))

exp_dir='pegasos'
n_init=10
for nmix in [2]:
    for l in ['27','9','3','1','0.333333','0.111111','0.037037','0.0123457','0.00411523']:
        W = np.load('exp/pegasos/warm_train_pegasos_w_%dC_%slambda_Prj.npy' % (nmix,l))
        W_meta = np.load('exp/%s/meta_%dC.npy' % (exp_dir,nmix))
        scores = np.dot(X,W.T)
        yhat39 = np.array(tuple( leehon_dict[k] for k in  W_meta[:,0][scores.argmax(1)] ))
        C = confusion_matrix(yhat39,y39)
        np.save('exp/%s/leehon39_warm_train_pegasos_dev_%dNinit_%dC_%slambda_Prj_confusion_matrix.npy' % (exp_dir,n_init,nmix,l),C)
        print exp_dir, nmix, np.sum(yhat39 == y39)/len(y39)
        open('exp/%s/leehon39_warm_train_pegasos_dev_%dNinit_%dC_%slambda_Prj_error_rate.txt' % (exp_dir,n_init,nmix,l),'w').write(str(np.sum(yhat39 == y39)/len(y39)))
        W = np.load('exp/pegasos/warm_train_pegasos_w_%dC_%slambda_NoPrj.npy'% (nmix,l))
        W_meta = np.load('exp/%s/meta_%dC.npy' % (exp_dir,nmix))
        scores = np.dot(X,W.T)
        yhat39 = np.array(tuple( leehon_dict[k] for k in  W_meta[:,0][scores.argmax(1)] ))
        C = confusion_matrix(yhat39,y39)
        np.save('exp/%s/leehon39_warm_train_pegasos_dev_%dNinit_%dC_%slambda_NoPrj_confusion_matrix.npy' % (exp_dir,n_init,nmix,l),C)
        print exp_dir, nmix, np.sum(yhat39 == y39)/len(y39)
        open('exp/%s/leehon39_warm_train_pegasos_dev_%dNinit_%dC_%slambda_NoPrj_error_rate.txt' % (exp_dir,n_init,nmix,l),'w').write(str(np.sum(yhat39 == y39)/len(y39)))


exp_dir='pegasos'
n_init=10
nmix=2
for wpath in ['warm_train_pegasos_w_iter3_280450niter_2C_9lambda_TruePrj_TrueRD.npy',
              'warm_train_pegasos_w_iter2_280450niter_2C_9lambda_TruePrj_TrueRD.npy',
              'warm_train_pegasos_w_iter1_280450niter_2C_9lambda_TruePrj_TrueRD.npy',
              'warm_train_pegasos_w_iter0_280450niter_2C_9lambda_TruePrj_TrueRD.npy']:
        W = np.load('exp/pegasos/%s' % wpath)
        W_meta = np.load('exp/%s/meta_%dC.npy' % (exp_dir,nmix))
        scores = np.dot(X,W.T)
        yhat39 = np.array(tuple( leehon_dict[k] for k in  W_meta[:,0][scores.argmax(1)] ))
        print exp_dir, nmix,wpath, np.sum(yhat39 == y39)/len(y39)
        open('exp/%s/leehon39_warm_train_pegasos_dev_%s_error_rate.txt' % (exp_dir,wpath),'w').write(str(np.sum(yhat39 == y39)/len(y39)))

exp_dir='pegasos'
n_init=10
nmix=2
for wpath in ['warm_train_pegasos_w_iter0_280450niter_2C_0.333333lambda_FalsePrj_TrueRD.npy',
              'warm_train_pegasos_w_iter1_280450niter_2C_0.333333lambda_FalsePrj_TrueRD.npy',
              'warm_train_pegasos_w_iter2_280450niter_2C_0.333333lambda_FalsePrj_TrueRD.npy',
              'warm_train_pegasos_w_iter3_280450niter_2C_0.333333lambda_FalsePrj_TrueRD.npy']:
        W = np.load('exp/pegasos/%s' % wpath)
        W_meta = np.load('exp/%s/meta_%dC.npy' % (exp_dir,nmix))
        scores = np.dot(X,W.T)
        yhat39 = np.array(tuple( leehon_dict[k] for k in  W_meta[:,0][scores.argmax(1)] ))
        print exp_dir, nmix,wpath, np.sum(yhat39 == y39)/len(y39)
        open('exp/%s/leehon39_warm_train_pegasos_dev_%s_error_rate.txt' % (exp_dir,wpath),'w').write(str(np.sum(yhat39 == y39)/len(y39)))

exp_dir='pegasos'
n_init=10
nmix=2
for wpath in ['warm_train_pegasos_w_iter1_280450niter_2C_1lambda_FalsePrj_TrueRD.npy',
              ]:
        W = np.load('exp/pegasos/%s' % wpath)
        W_meta = np.load('exp/%s/meta_%dC.npy' % (exp_dir,nmix))
        scores = np.dot(X,W.T)
        yhat39 = np.array(tuple( leehon_dict[k] for k in  W_meta[:,0][scores.argmax(1)] ))
        print exp_dir, nmix,wpath, np.sum(yhat39 == y39)/len(y39)
        open('exp/%s/leehon39_warm_train_pegasos_dev_%s_error_rate.txt' % (exp_dir,wpath),'w').write(str(np.sum(yhat39 == y39)/len(y39)))

exp_dir='pegasos'
n_init=10
nmix=21
for wpath in ['warm_train_pegasos_w_iter1_280450niter_21C_9lambda_TruePrj_FalseRD.npy',
              'warm_train_pegasos_w_iter0_280450niter_21C_9lambda_TruePrj_FalseRD.npy']:
        W = np.load('exp/pegasos/%s' % wpath)
        W_meta = np.load('exp/%s/meta_%dC.npy' % (exp_dir,nmix))
        scores = np.dot(X,W.T)
        yhat39 = np.array(tuple( leehon_dict[k] for k in  W_meta[:,0][scores.argmax(1)] ))
        print exp_dir, nmix,wpath, np.sum(yhat39 == y39)/len(y39)
        open('exp/%s/leehon39_dev_%s_error_rate.txt' % (exp_dir,wpath),'w').write(str(np.sum(yhat39 == y39)/len(y39)))

exp_dir='pegasos'
n_init=10
nmix=21
for wpath in ['warm_train_pegasos_w_iter13_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter12_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter11_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter10_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter9_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter8_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter7_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter6_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter5_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter4_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter3_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter2_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
'warm_train_pegasos_w_iter1_560900niter_21C_9lambda_FalsePrj_FalseRD.npy',
              'warm_train_pegasos_w_iter0_560900niter_21C_9lambda_FalsePrj_FalseRD.npy']:
        W = np.load('exp/pegasos/%s' % wpath)
        W_meta = np.load('exp/%s/meta_%dC.npy' % (exp_dir,nmix))
        scores = np.dot(X,W.T)
        yhat39 = np.array(tuple( leehon_dict[k] for k in  W_meta[:,0][scores.argmax(1)] ))
        print exp_dir, nmix,wpath, np.sum(yhat39 == y39)/len(y39)
        open('exp/%s/leehon39_dev_%s_error_rate.txt' % (exp_dir,wpath),'w').write(str(np.sum(yhat39 == y39)/len(y39)))




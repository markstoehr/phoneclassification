from __future__ import division
from amitgroup.stats import bernoullimm
import numpy as np

rootdir='/var/tmp/stoehr/phoneclassification'
confdir='%s/conf'%rootdir
datadir='%s/data/local/data' % rootdir
expdir='%s/exp/bmm_hierarchy' % rootdir

leehon=np.loadtxt('%s/phones.48-39' % confdir,dtype=str)

nobs = np.zeros(len(leehon[:,0]))
for phone_id, phone in enumerate(leehon[:,0]):
    X = np.load('%s/%s_train_examples.npy' % (datadir,phone))
    nobs[phone_id] = X.shape[0]
    dim = np.prod(X.shape[1:])
    print dim

X = np.zeros((nobs.sum(),dim),dtype=np.uint8)
y = np.zeros(nobs.sum(),dtype=int)
curobs=0
for phone_id, phone in enumerate(leehon[:,0]):
    X[curobs:curobs+nobs[phone_id]] = np.load('%s/%s_train_examples.npy' % (datadir,phone)).reshape(nobs[phone_id],dim)
    y[curobs:curobs+nobs[phone_id]] = phone_id
    curobs += nobs[phone_id]


bmm=bernoullimm.BernoulliMM(n_components=9,n_iter=100,n_init=5,verbose=True,float_type=np.float32,blocksize=8000)
bmm.fit(X)

np.save('%s/all_class_means_9c.npy' % expdir,bmm.means_)
np.save('%s/all_class_weights_9.npy' % expdir,bmm.weights_)
class_clusters = bmm.cluster_underlying_data(class_ids,X)
np.save('%s/all_class_clusters_9c.npy' % expdir,class_clusters)
np.save('%s/all_class_responsibilities_9c.npy' % expdir,responsibilities.T)

class_ids = np.zeros((y.shape[0],len(leehon[:,0])),dtype=np.uint8)
class_ids[np.arange(len(y)),y] = 1

for n_components in [15,20,25,35,48,60]:
    bmm=bernoullimm.BernoulliMM(n_components=n_components,n_iter=100,n_init=5,verbose=True,float_type=np.float32,blocksize=8000)
    bmm.fit(X)
    np.save('%s/all_class_means_%dc.npy' % (expdir,n_components),bmm.means_)
    np.save('%s/all_class_weights_%dc.npy' % (expdir,n_components),bmm.weights_)
    responsibilities=bmm.predict_proba(X)
    np.save('%s/all_class_responsibilities_%dc.npy' % (expdir,n_components), responsibilities)
    class_clusters = bmm.cluster_underlying_data(class_ids,X)
    np.save('%s/all_class_clusters_%dc.npy' % (expdir,n_components),class_clusters)








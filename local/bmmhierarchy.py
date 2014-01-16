from __future__ import division
from src import bernoullimm
import numpy as np

# rootdir='/home/mark/Research/phoneclassification'
rootdir='/var/tmp/stoehr/phoneclassification'

confdir='%s/conf'%rootdir
datadir='%s/data/local/data' % rootdir
expdir='%s/exp/bmm_hierarchy' % rootdir

leehon=np.loadtxt('%s/phones.48-39' % confdir,dtype=str)
phones39 = np.unique(np.sort(leehon[:,1]))
phones39_dict = dict( (v,i) for i,v in enumerate(phones39))
phones48_dict = dict( (v,i) for i,v in enumerate(leehon[:,0]))
leehon_dict = dict( (phones48_dict[p],
                     phones39_dict[q]) for p,q in leehon)


nobs = np.zeros(len(leehon[:,0]))
for phone_id, phone in enumerate(leehon[:,0]):
    X = np.load('%s/%s_train_examples.npy' % (datadir,phone))
    nobs[phone_id] = X.shape[0]
    dim = np.prod(X.shape[1:])
    print dim

X = np.zeros((nobs.sum(),dim),dtype=np.uint8)
y = np.zeros(nobs.sum(),dtype=int)
y39 = np.zeros(nobs.sum(),dtype=int)
curobs=0
for phone_id, phone in enumerate(leehon[:,0]):
    X[curobs:curobs+nobs[phone_id]] = np.load('%s/%s_train_examples.npy' % (datadir,phone)).reshape(nobs[phone_id],dim)
    y[curobs:curobs+nobs[phone_id]] = phone_id
    y39[curobs:curobs+nobs[phone_id]] = leehon_dict[phone_id]
    curobs += nobs[phone_id]


means_ = np.load('%s/all_class_means_9c.npy' % expdir)
weights_ = np.load('%s/all_class_weights_9c.npy' % expdir)

bmm=bernoullimm.BernoulliMM(n_components=9,n_iter=100,n_init=1,verbose=True,float_type=np.float32,blocksize=8000,init_params='')
bmm.means_ = means_
bmm.weights_ = weights_
bmm.fitthresholdlabels(X,y39,threshold=75)

np.save('%s/layer1_class_means_9c.npy' % expdir,bmm.means_)
np.save('%s/layer1_class_weights_9c.npy' % expdir,bmm.weights_)
class_clusters = bmm.cluster_underlying_data(class_ids,X)
np.save('%s/layer1_class_clusters_9c.npy' % expdir,class_clusters)
responsibilities=bmm.predict_proba(X)
np.save('%s/layer1_class_responsibilities_9c.npy' % expdir,responsibilities)


responsibilities = np.load('%s/layer1_class_responsibilities_9c.npy' % expdir)
n_components=9
phoneclass_bclass_counts = np.zeros((len(leehon),n_components),dtype=int)
classifier_tree = np.zeros(phoneclass_bclass_counts.shape,dtype=np.uint8)
for pclass_id in xrange(len(leehon)):
    phoneclass_bclass_counts[pclass_id] = responsibilities[y==pclass_id].sum(0)
    classifier_tree[pclass_id] = ((phoneclass_bclass_counts[pclass_id]/30).astype(int) > 0 ).astype(np.uint8)

np.save('%s/layer1_classifier_tree9c.npy' % expdir,classifier_tree.T)
    
for phone_id, phone in enumerate(leehon[:,0]):
    print phone_id
    for component_id in xrange(n_components):
        print component_id
        layer_n_components = min(int(phoneclass_bclass_counts[phone_id,component_id]/30),6)
        if layer_n_components == 0: continue
        bmm=bernoullimm.BernoulliMM(n_components=layer_n_components,n_iter=100,n_init=5,verbose=True,float_type=np.float32,init_params='mw')
        bmm.fit(X[
            (responsibilities[:,component_id] > .9)
            *(y==phone_id)])
        np.save('%s/layer2_%s_%d_%d_means.npy' % (expdir,phone,phone_id,component_id),bmm.means_)
        np.save('%s/layer2_%s_%d_%d_weights.npy' % (expdir,phone,phone_id,component_id),bmm.weights_)


class_ids = np.zeros((y.shape[0],len(leehon[:,0])),dtype=np.uint8)
class_ids[np.arange(len(y)),y] = 1

class_clusters = bmm.cluster_underlying_data(class_ids,X)





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








freqs = (phoneclass_bclass_counts + .5) / np.lib.stride_tricks.as_strided(phoneclass_bclass_counts.sum(1),phoneclass_bclass_counts.shape,strides=(8,0))
log_freqs = np.log(freqs)
KLdivs = np.zeros((len(leehon),len(leehon)))

for pclass_id1 in xrange(len(leehon)):
    for pclass_id2 in xrange(len(leehon)):
        KLdivs[pclass_id1,pclass_id2] = np.sum(freqs[pclass_id1]*(log_freqs[pclass_id1] - log_freqs[pclass_id2]))

import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering
import matplotlib.pyplot as plt
import matplotlib.cm as cm

KLdivs_sparse=np.exp(-KLdivs)
threshold=.6
KLdivs_sparse[KLdivs_sparse < threshold] = 0
G = nx.Graph(KLdivs_sparse)
rcm = list(reverse_cuthill_mckee_ordering(G))

title='KL D(p||q) map 9 classes'
figsize=6
plt.close('all')
plt.figure(figsize=(figsize,figsize))
plt.imshow(-(1+np.clip((np.exp(-KLdivs[rcm]).T)[rcm].T,0,2)),
               interpolation='nearest',
               cmap='hot')

plt.grid()
plt.xticks(np.arange(len(rcm)),leehon[:,0][rcm])
plt.yticks(np.arange(len(rcm)),leehon[:,0][rcm])
plt.xlabel('p')
plt.ylabel('q')
plt.title(title)
plt.savefig('%s/rcm_kl_div_9c.png' % expdir,bbox_inches='tight')

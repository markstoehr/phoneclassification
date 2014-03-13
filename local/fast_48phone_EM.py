from __future__ import division
from phoneclassification.confusion_matrix import confusion_matrix
import numpy as np
import argparse,collections
from phoneclassification._fast_EM import EM, e_step, m_step, sparse_dotmm
from phoneclassification.binary_sgd import binary_to_bsparse, add_final_one


"""
Extract the set of data associated with a set of phones
and give a label set, also initialize the components using the
basic components from the model
"""

def get_use_phns_row_ids(W_meta,use_phns,phones_dict):
    row_ids = []
    for phn_id, phn in enumerate(use_phns):
        phn_row_ids = np.where(W_meta[:,0]==phones_dict[phn])[0]
        row_ids.extend(phn_row_ids)
        W_meta[phn_row_ids,0] = phn_id

    return W_meta, np.array(row_ids)

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

parser.add_argument('--use_sparse_suffix',default=None,
                    type=str,help='If not included then we do not assume a sparse save structure for the data otherwise this is the suffix for where the data are stored in sparse format')
parser.add_argument('--dev_sparse_suffix',default=None,
                    type=str,help='If not included then we do not assume a sparse save structure for the data otherwise this is the suffix for where the data are stored in sparse format')
parser.add_argument('--in_prefix',type=str,help='prefix for path the data were saved to')
parser.add_argument('--in_suffix',type=str,help='suffix for path the data were saved to')

parser.add_argument('--out_prefix',type=str,help='prefix for path to save the output to')
parser.add_argument('--out_suffix',type=str,help='suffix for path to save the output to')
parser.add_argument('--total_iter',type=np.intc,help='Number of iterations to run this for')
parser.add_argument('--total_init',type=np.intc,help='Number of initializations to use in estimating the models')
parser.add_argument('--min_counts',type=np.intc,help='Minimum number of examples for each component')
parser.add_argument('--tol',type=float,help='Convergence criterion')
parser.add_argument('--ncomponents',type=np.intc,help='Maximum number of components per model')
parser.add_argument('--simple_data_load',action='store_true',help='whether to use simple data loading')
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
leehon_dict_array = np.zeros(48,dtype=int)
for k,v in leehon_dict.items():
    leehon_dict_array[k] = int(v)


leehon_phn_dict = dict( (p,q) for p,q in leehon)

leehon39to48 = collections.defaultdict(list)

for phn in leehon[:,0]:
    leehon39to48[leehon_phn_dict[phn]].append(phn)

use_phns39 = list(phones39[:])
use_phns48 = leehon[:,0]

if args.simple_data_load:
    all_feature_ids_train = np.load('%s/all_feature_ids_train_%s'
                                % (args.in_prefix,args.in_suffix))
    
    example_nnz_train = np.load('%s/example_nnz_train_%s'
                                % (args.in_prefix,args.in_suffix))

    rowstartidx_train = np.zeros(len(example_nnz_train)+1,dtype=np.intc)
    rowstartidx_train[1:] = np.cumsum(example_nnz_train)
    y_train = np.load('%s/y_train_%s'
                      % (args.in_prefix,args.in_suffix))

    feature_dim = np.load('%s/example_dim_train_%s' % (args.in_prefix,args.in_suffix))
    dim = np.intc(np.prod(feature_dim))
    
    n_train_data = example_nnz_train.shape[0]

    all_feature_ids_dev = np.load('%s/all_feature_ids_dev_%s'
                                % (args.in_prefix,args.in_suffix))

    example_nnz_dev = np.load('%s/example_nnz_dev_%s'
                                % (args.in_prefix,args.in_suffix))
    rowstartidx_dev = np.zeros(len(example_nnz_dev)+1,dtype=np.intc)
    rowstartidx_dev[1:] = np.cumsum(example_nnz_dev)
    y_dev = np.load('%s/y_dev_%s'
                    % (args.in_prefix,args.in_suffix))
    n_dev_data = example_nnz_dev.shape[0]

    all_feature_ids_dev, example_nnz_dev,rowstartidx_dev = add_final_one(all_feature_ids_dev, example_nnz_dev,rowstartidx_dev,dim)
    y_dev39 = np.array([ leehon_dict[phone_id] for phone_id in y_dev]).astype(np.int16)
    dev_accuracy = lambda W : np.sum(leehon_dict_array[weights_classes[sparse_dotmm(all_feature_ids_dev,example_nnz_dev,rowstartidx_dev,W.ravel().copy(),n_dev_data,W.shape[1],W.shape[0]).argmax(1)]] == y_dev39)/float(len(y_dev39))



elif args.use_sparse_suffix is None:
    nobs = np.zeros(len(use_phns48))
    for phone_id, phone in enumerate(use_phns48):    
        X = np.load('%s/%s_train_examples.npy' % (datadir,phone))
        nobs[phone_id] = X.shape[0]
        dim = np.prod(X.shape[1:])
        print dim

    X = np.ones((nobs.sum(),dim),dtype=np.uint8)
    y = np.zeros(nobs.sum(),dtype=int)
    y39 = np.zeros(nobs.sum(),dtype=int)
    curobs=0
    for phone_id, phone in enumerate(use_phns48):
        X[curobs:curobs+nobs[phone_id]] = np.load('%s/%s_train_examples.npy' % (datadir,phone)).reshape(nobs[phone_id],dim)
        y[curobs:curobs+nobs[phone_id]] = phone_id
        y39[curobs:curobs+nobs[phone_id]] = leehon_dict[phone_id]
        curobs += nobs[phone_id]
    
    
    nobs = np.zeros(len(use_phns48))
    for phone_id, phone in enumerate(use_phns48):    
        X_test = np.load('%s/%s_dev_examples.npy' % (datadir,phone))
        nobs[phone_id] = X_test.shape[0]
        dim = np.prod(X_test.shape[1:])
        print dim


    X_test = np.ones((nobs.sum(),dim+1),dtype=np.uint8)
    y_test = np.zeros(nobs.sum(),dtype=int)
    y_test39 = np.zeros(nobs.sum(),dtype=int)
    curobs=0
    for phone_id, phone in enumerate(use_phns48):
        X_test[curobs:curobs+nobs[phone_id],:-1] = np.load('%s/%s_dev_examples.npy' % (datadir,phone)).reshape(nobs[phone_id],dim)
        y_test[curobs:curobs+nobs[phone_id]] = phone_id
        y_test39[curobs:curobs+nobs[phone_id]] = leehon_dict[phone_id]
        curobs += nobs[phone_id]

    feature_ind, rownnz, rowstartidx = binary_to_bsparse(X)
    feature_ind = feature_ind[:,1].copy()
    y = y.astype(np.int16) 
    del X
    test_accuracy = lambda W : np.sum(leehon_dict_array[weights_classes[np.dot(X_test,W.T).argmax(1)]] == y_test39)/float(len(y_test39))
    

else:
    feature_ind = np.load('%sX_indices_%s' % (args.data_dir,
                                              args.use_sparse_suffix),
                          )
    rownnz = np.load('%sX_rownnz_%s' % (args.data_dir,
                                              args.use_sparse_suffix),
                          )
    dim = np.intc(np.prod(np.load('%sdim_%s' % (args.data_dir, args.use_sparse_suffix))))

    X_n_rows = rownnz.shape[0]
    
    rowstartidx = np.load('%sX_rowstartidx_%s' % (args.data_dir,
                                              args.use_sparse_suffix),
                          )

    y = np.load('%sy_%s' % (args.data_dir,
                                              args.use_sparse_suffix),
                          ).astype(np.int16)

    feature_ind_test = np.load('%sX_indices_%s' % (args.data_dir,
                                              args.dev_sparse_suffix),
                          )
    rownnz_test = np.load('%sX_rownnz_%s' % (args.data_dir,
                                              args.dev_sparse_suffix),
                          )
    rowstartidx_test = np.load('%sX_rowstartidx_%s' % (args.data_dir,
                                              args.dev_sparse_suffix),
                           )
    feature_ind_test, rownnz_test,rowstartidx_test = add_final_one(feature_ind_test,rownnz_test,rowstartidx_test,dim)
    
    y_test = np.load('%sy_%s' % (args.data_dir,
                                              args.dev_sparse_suffix),
                          ).astype(np.int16)
    X_n_rows_test = y_test.shape[0]
    y_test39 = np.array([ leehon_dict[phone_id] for phone_id in y_test]).astype(np.int16)
    test_accuracy = lambda W : np.sum(leehon_dict_array[weights_classes[sparse_dotmm(feature_ind_test,rownnz_test,rowstartidx_test,W.ravel().copy(),X_n_rows_test,W.shape[1],W.shape[0]).argmax(1)]] == y_test39)/float(len(y_test39))


max_n_classifiers = args.ncomponents * 48
classifier_id = 0
for phn_id, phn in enumerate(leehon[:,0]):
    print "Working on phone %s which has id %d" % (phn, phn_id)
    print "classifier_id = %d" % classifier_id
    
    phn_n_data = (y_train == phn_id).sum()
    phn_rownnz = example_nnz_train[y_train==phn_id].copy()
    phn_start_idx = np.where(y_train==phn_id)[0].min()
    phn_end_idx = np.where(y_train==phn_id)[0].max()+1
    if (phn_end_idx - phn_start_idx) != len(phn_rownnz):
        import pdb; pdb.set_trace()
        
    phn_rowstartidx = rowstartidx_train[phn_start_idx:phn_end_idx+1].copy()
    phn_feature_ind = all_feature_ids_train[phn_rowstartidx[0]:phn_rowstartidx[-1]].copy()
    phn_rowstartidx -= phn_rowstartidx[0]
    
    converged = False
    cur_ncomponents = args.ncomponents
    
    if phn_id == 0:
        avgs = np.zeros((max_n_classifiers,
                         dim) )
        counts = np.zeros(max_n_classifiers
        )
        # will keep track of which average belongs to which
        # phone and mixture component--this allows us to
        # drop mixture components if they are potentially
        # not helping
        all_weights = np.zeros(max_n_classifiers,dtype=float)
        meta = np.zeros((max_n_classifiers
                             ,2),dtype=int)

        
    
    n_init = 0
    tol = float(args.tol)
    total_iter = np.intc(args.total_iter)
    while n_init < args.total_init:
        A = np.zeros((phn_n_data,cur_ncomponents),dtype=float)
        A[np.arange(phn_n_data),np.random.randint(cur_ncomponents,size=phn_n_data)] = 1
        A = A.reshape(A.size)
        P = np.zeros(dim*cur_ncomponents,dtype=float)
        weights = np.zeros(cur_ncomponents,dtype=float)

        # m_step(phn_feature_ind, phn_rownnz, phn_rowstartidx,
        #        P,weights, A, phn_n_data, dim, cur_ncomponents)
        # import pdb; pdb.set_trace() 
        P,weights, A, loglikelihood = EM(phn_feature_ind, phn_rownnz, phn_rowstartidx,
           phn_n_data,dim,cur_ncomponents,tol, total_iter,
                          A)
        A = A.reshape(phn_n_data,cur_ncomponents)
        P = P.reshape(cur_ncomponents, dim)
        component_counts = A.sum(0)
        good_components = component_counts >= args.min_counts
        n_good = good_components.sum()
        while np.any(component_counts < args.min_counts):           
            good_components = component_counts >= args.min_counts
            n_good = good_components.sum()
            P = P[good_components]
            weights = weights[good_components]
            A = np.zeros((phn_n_data,n_good),dtype=float)
            A = A.reshape(A.size)
            P = P.reshape(P.size)


            likelihood = e_step(phn_feature_ind, 
                   phn_rownnz, 
                   phn_rowstartidx,
                   P, 
                   weights, 
                   A, phn_n_data, dim, n_good )

            P,weights, A, loglikelihood = EM(phn_feature_ind, phn_rownnz, phn_rowstartidx,
                                             
                                             phn_n_data,dim,n_good,args.tol, args.total_iter,
                                             A)
            A = A.reshape(phn_n_data,n_good)
            P = P.reshape(n_good, dim)
            component_counts = A.sum(0)
        
        if n_init == 0:
            bestP = P.copy()
            bestweights = weights.copy()
            best_ll = loglikelihood
            n_use_components = n_good
        elif loglikelihood > best_ll:
            print "Updated best loglikelihood to : %g " % loglikelihood
            bestP = P.copy()
            bestweights = weights.copy()
            best_ll = loglikelihood
            n_use_components = n_good
            
        n_init += 1

    # add the components
    avgs[classifier_id:classifier_id + n_use_components] = bestP[:]
    all_weights[classifier_id:classifier_id + n_use_components] = bestweights[:]
    meta[classifier_id:classifier_id+n_use_components,0] = phn_id
    meta[classifier_id:classifier_id+n_use_components,1] = np.arange(n_use_components)
    
    classifier_id += n_use_components

print "Total of %d models" % classifier_id
np.save('%s/avgs_%s' % (args.out_prefix, args.out_suffix),
            avgs[:classifier_id])

np.save('%s/weights_%s' % (args.out_prefix, args.out_suffix),
            weights[:classifier_id])

np.save('%s/meta_%s' % (args.out_prefix, args.out_suffix),
            meta[:classifier_id])


# now we test the model to see what happens
avgs = avgs.reshape(avgs.shape[0],
                    dim)
W = np.zeros((len(avgs),dim+1))
W[:,:-1] = np.log(avgs) - np.log(1-avgs)
W[:,-1] = np.log(1-avgs).sum(-1)
W_meta = meta.astype(np.intc)



# need to construct W_meta39 to use 39 labels
W_meta39 = get_reduced_meta(W_meta,leehon_dict).astype(np.intc)
# now we get the use_phns39 row ids



weights = W.ravel().copy()
weights_classes = W_meta[:,0].copy()
weights_components = W_meta[:,1].copy()
sorted_component_ids = np.argsort(weights_components,kind='mergesort')
sorted_components = weights_components[sorted_component_ids]
sorted_weights_classes = weights_classes[sorted_component_ids]
stable_sorted_weights_classes_ids = np.argsort(sorted_weights_classes,kind='mergesort')
weights_classes = sorted_weights_classes[stable_sorted_weights_classes_ids]
weights_components = sorted_components[stable_sorted_weights_classes_ids]

W = W[sorted_component_ids][stable_sorted_weights_classes_ids]

n_classes = 48
print "n_classes=%d" % n_classes


accuracy = dev_accuracy(W)
print "test accuracy = %g" % accuracy

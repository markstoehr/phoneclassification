from __future__ import division
import numpy as np
from scipy.spatial.distance import pdist, squareform

def perform_update_single_example(YX,T,l):
    use_example_ids = np.random.randint(0,YX.shape[0],size=T)
    w = np.zeros((YX.shape[0]+1,) + YX.shape[1:])
    for t, example_id in enumerate(use_example_ids):
        yx = YX[example_id]
        eta_t = 1/(l * (t+1))
        score = np.dot(w[t],yx)
        if score < 1:
            w[t+1] = (1-eta_t*l)*w[t] + eta_t * yx
        else:
            w[t+1] = (1- eta_t*l)*w[t]

    return w

def perform_batch_updates(YX,T,l,k,w_init=None,v=False):
    k = min(YX.shape[0],k)
    use_example_ids = np.random.randint(0,YX.shape[0],size=(T,k))
    w = np.zeros((T+1,) + YX.shape[1:])
    if w_init is not None:
        w[0] = w_init
    for t, example_ids in enumerate(use_example_ids):
        if v:
            scores = np.dot(YX,w[t])
            loss = np.maximum(1-scores,0).sum()
            print t,loss
            A_plus = YX[example_ids][np.nonzero(scores[example_ids] < 1)[0]]
        else:
            batch = YX[example_ids]
            A_plus = batch[np.nonzero(np.dot(batch,w[t]) < 1)[0]]
        eta_t = 1/(l*(t+1))
        w[t+1] = (1 - eta_t*l)*w[t] + eta_t/k *A_plus.sum(0)

    return w

def multiclass(Y,X,T,l,k,W_init,W_classes,start_t=100,v=False,loss='hinge',
               do_projection=True,verbose=True):
    """
    Parameters
    ----------
    Y :   (n_samples,)
        Y classes
    X :   (n_samples, n_features)
        Data 
    T :  int
       Number of rounds over which to perform
       optimization

    l : float
       lambda regularization parameter
    k : int
       batch size
    W_init : (n_classifiers,n_features) 
       Initializaiton for W -- required
    W_meta : (n_classifiers, 2)
       class identities and mixture component
       identity for each of the classifiers in
       W_init

    In each round we get the scores
    then we construct a class-mask over the scores
    and we map all the scores to a lower quantity

    """
    k = min(X.shape[0],k)
    use_example_ids = np.random.randint(0,X.shape[0],size=(T,)) 
    n_classes = max(W_classes[:,0].max()+1,Y.max()+1)
    class_masks = np.zeros((n_classes,W_init.shape[0]),dtype=bool)
    for y in xrange(n_classes):
        class_masks[y] = W_classes[:,0] == y

    oneoversqrtlambda = 1/np.sqrt(l)

    for t, example_id in enumerate(use_example_ids):
        if t % 1000 == 0:
            if verbose:
                print t
            
        y = Y[example_id]
        scores = np.dot(W_init,X[example_id])
        add_quantity = scores.max() - scores.min() +1
        true_class_best_score_id = np.argmax(scores + add_quantity* class_masks[y])

        true_class_best_score = scores[true_class_best_score_id]
        not_class_best_score_id = np.argmax(scores - add_quantity*class_masks[y])
        not_class_best_score = scores[not_class_best_score_id]

        scaling = 1./(t+1.+start_t)
        W_init -=  scaling * W_init
        # check the loss
        if (loss=='hinge' and true_class_best_score - not_class_best_score < 1) or (loss=='ramp' and abs(true_class_best_score - not_class_best_score) < 1):
            eta = scaling/l
            W_init[true_class_best_score_id] += eta*X[example_id]
            W_init[not_class_best_score_id] -= eta*X[example_id]

        if do_projection:
            if type(do_projection) == int:
                if t % do_projection != 0: continue
            W_init_norm = np.linalg.norm(W_init)
            if W_init_norm == 0:
                return W_init
            scaling = oneoversqrtlambda/W_init_norm
            W_init *= min(1,scaling)
            
    return W_init


def multiclass_regularize_diffs(Y,X,T,l,k,W_init,W_classes,v=False,loss='hinge',
                                do_projection=True,verbose=True):
    """
    Parameters
    ----------
    Y :   (n_samples,)
        Y classes
    X :   (n_samples, n_features)
        Data 
    T :  int
       Number of rounds over which to perform
       optimization

    l : float
       lambda regularization parameter
    k : int
       batch size
    W_init : (n_classifiers,n_features) 
       Initializaiton for W -- required
    W_meta : (n_classifiers, 2)
       class identities and mixture component
       identity for each of the classifiers in
       W_init

    In each round we get the scores
    then we construct a class-mask over the scores
    and we map all the scores to a lower quantity

    """
    k = min(X.shape[0],k)

    use_example_ids = np.random.randint(0,X.shape[0],size=(T,))
    n_classes = max(W_classes[:,0].max()+1,Y.max()+1)
    class_masks = np.zeros((n_classes,W_init.shape[0]),dtype=bool)
    for y in xrange(n_classes):
        class_masks[y] = W_classes[:,0] == y

    oneoversqrtlambda = 1/np.sqrt(l)
    
    # get the number of classifiers
    M = W_init.shape[0]
    for t, example_id in enumerate(use_example_ids):
        if t % 1000 == 0:
            if verbose:
                print t

        y = Y[example_id]
        scores = np.dot(W_init,X[example_id])
        add_quantity = scores.max() - scores.min() +1
        true_class_best_score_id = np.argmax(scores + add_quantity* class_masks[y])

        true_class_best_score = scores[true_class_best_score_id]
        not_class_best_score_id = np.argmax(scores - add_quantity*class_masks[y])
        not_class_best_score = scores[not_class_best_score_id]

        scaling = 1/(t+1.+M)
        W_init -=  scaling * (M* W_init - W_init.sum(0))
        # check the loss
        if (loss=='hinge' and true_class_best_score - not_class_best_score < 1) or (loss=='ramp' and abs(true_class_best_score - not_class_best_score) < 1):
            eta = scaling/l
            W_init[true_class_best_score_id] += eta*X[example_id]
            W_init[not_class_best_score_id] -= eta*X[example_id]

        if do_projection:
            if type(do_projection)==int:
                if t % do_projection != 0: continue
            sqW_init_norm = np.sqrt(pdist(W_init,'sqeuclidean').sum())
            if sqW_init_norm > oneoversqrtlambda:
                wl = oneoversqrtlambda/sqW_init_norm
                wl_top = sqW_init_norm/oneoversqrtlambda
                W_init = wl * W_init + (wl_top-1)*wl/M * W_init.sum(0)
            
    return W_init

    
    

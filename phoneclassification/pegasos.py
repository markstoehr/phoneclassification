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

def multiclass(Y,X,T,l,W,start_t=1,loss_computation=0,
               return_avg_W=True,return_loss=True,verbose=False,loss='hinge',
               do_projection=False):
    """
    A multiclass implementation of the Pegasos
    similar to the one presented by Zhuang Wang et al. 2010
    ICML

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

    W : (n_classifiers,n_features)
       Initializaiton for W -- required
       the assumption is that W[c] is the
       classifier for class c -- i.e. we work
       from the rows as classifiers

    return_avg_W : bool
       If True then we return the W produced as the average
       of the updates

    start_t : int
       Starting round -- in the case of a warm-start one
       may not wish to begin with round 1 since, presumably
       the first few rounds have already been passed

    loss_computation : int
       If 0 then we do not compute the loss over the data set while
       running the algorithm (which is good if the dataset is large),
       otherwise if loss_computation > 0 we compute it every
       loss_computation rounds.

    In each round we get the scores
    then we construct a class-mask over the scores
    and we map all the scores to a lower quantity
    """
    use_example_ids = np.random.randint(0,X.shape[0],size=(T,))
    n_classes, n_features = W.shape
    # class masks makes finding best non-class happen more easily
    class_masks = np.eye(n_classes)
    init_W = W.copy()
    avg_W_update = np.zeros(W.shape)
    new_W_delta = W.copy()

    oneoversqrtlambda = 1/np.sqrt(l)
    loss_list = []
    avg_W_loss_list = []
    for t, example_id in enumerate(use_example_ids):
        if t % 1000 == 0:
            if verbose: print t

        y = Y[example_id]
        scores = np.dot(W,X[example_id])
        add_quantity = scores.max() - scores.min() + 1
        true_class_best_score = scores[y]
        not_class_best_score_id = np.argmax( scores - add_quantity * class_masks[y])
        not_class_best_score = scores[not_class_best_score_id]

        scaling = 1./max(1,t+start_t)
        new_W_delta = - scaling * W
        if (loss=='hinge' and true_class_best_score - not_class_best_score < 1) or (loss=='ramp' and abs(true_class_best_score - not_class_best_score) < 1):
            eta = scaling/l
            new_W_delta[y] += eta*X[example_id]
            new_W_delta[not_class_best_score_id] -= eta*X[example_id]
        
        if return_avg_W:
            avg_W_update += (new_W_delta - avg_W_update)/(t+1)

        W += new_W_delta
            
        if do_projection:
            if type(do_projection) == int:
                if t % do_projection != 0: continue

            W_norm = np.linalg.norm(W)
            if W_norm == 0:
                return_tuple = (W,)
                if return_avg_W:
                    return_tuple += (init_W + avg_W_update,)
                if return_loss:
                    return_tuple += (loss_list,)
                if return_avg_W and return_loss:
                    return_tuple += (avg_W_loss_list,)
                return return_tuple
            scaling = oneoversqrtlambda/W_norm
            W *= min(1,scaling)

        
        # compute the zero-one loss to check for convergence
        if loss_computation > 0 and (t % loss_computation == 0):
            loss_list.append((t,(np.dot(X,W.T).argmax(1) == Y).sum() / X.shape[0]))
            print "round %d: loss=%g" % (t,loss_list[-1][-1] )

            if return_avg_W:
                avg_W_loss_list.append((t, (np.dot(X,(init_W+avg_W_update).T).argmax(1) == Y).sum() / X.shape[0]))
                print "round %d: loss avgW=%g" % (t, avg_W_loss_list[-1][-1])


    return_tuple = (W,)
    if return_avg_W:
        return_tuple += (init_W + avg_W_update,)
    if return_loss:
        return_tuple += (loss_list,)
    if return_avg_W and return_loss:
        return_tuple += (avg_W_loss_list,)
    return return_tuple




def multiclass_minibatch(Y,X,T,l,k,W,start_t=1,loss_computation=0,
               return_avg_W=True,verbose=False,loss='hinge',
               do_projection=False):
    """
    A multiclass implementation of the Pegasos
    similar to the one presented by Zhuang Wang 2010

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
    W : (n_features,n_classifiers)
       Initializaiton for W -- required
       the assumption is that W[:,c] is the
       classifier for class c -- i.e. we work
       from the columns which means that dot
       products will work quickly with large
       chunks of the data

    return_avg_W : bool
       If True then we return the W produced as the average
       of the updates

    start_t : int
       Starting round -- in the case of a warm-start one
       may not wish to begin with round 1 since, presumably
       the first few rounds have already been passed

    loss_computation : int
       If 0 then we do not compute the loss over the data set while
       running the algorithm (which is good if the dataset is large),
       otherwise if loss_computation > 0 we compute it every
       loss_computation rounds.

    In each round we get the scores
    then we construct a class-mask over the scores
    and we map all the scores to a lower quantity
    """
    k = min(X.shape[0],k)
    use_example_ids = np.random.randint(0,X.shape[0],size=(T,))
    n_classes = max(W_classes[:,0].max()+1,Y.max()+1)
    class_masks = np.zeros((n_classes, W_init.shape[0]))

def multiclass_multicomponent(Y,X,T,l,W,W_classes,start_t=100,v=False,loss='hinge',loss_computation=0,
               return_avg_W=True,return_loss=True,
               do_projection=True,verbose=True):
    """
    The update is declared weird since it does not actually use
    the subgradient but another value

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
    W : (n_classifiers,n_features)
       Initializaiton for W -- required
    W_meta : (n_classifiers, 2)
       class identities and mixture component
       identity for each of the classifiers in
       W_init

    return_avg_W : bool
       If True then we return the W produced as the average
       of the updates

    start_t : int
       Starting round -- in the case of a warm-start one
       may not wish to begin with round 1 since, presumably
       the first few rounds have already been passed

    loss_computation : int
       If 0 then we do not compute the loss over the data set while
       running the algorithm (which is good if the dataset is large),
       otherwise if loss_computation > 0 we compute it every
       loss_computation rounds.

    In each round we get the scores
    then we construct a class-mask over the scores
    and we map all the scores to a lower quantity

    """
    use_example_ids = np.random.randint(0,X.shape[0],size=(T,))
    n_classes = max(W_classes[:,0].max()+1,Y.max()+1)
    class_masks = np.zeros((n_classes,W.shape[0]),dtype=bool)
    for y in xrange(n_classes):
        class_masks[y] = W_classes[:,0] == y

    init_W = W.copy()
    avg_W_update = np.zeros(W.shape)
    new_W_delta = W.copy()

    oneoversqrtlambda = 1/np.sqrt(l)
    loss_list = []
    avg_W_loss_list = []

    for t, example_id in enumerate(use_example_ids):
        if t % 1000 == 0:
            if verbose:
                print t

        y = Y[example_id]
        scores = np.dot(W,X[example_id])
        add_quantity = scores.max() - scores.min() +1
        true_class_best_score_id = np.argmax(scores + add_quantity* class_masks[y])

        true_class_best_score = scores[true_class_best_score_id]
        not_class_best_score_id = np.argmax(scores - add_quantity*class_masks[y])
        not_class_best_score = scores[not_class_best_score_id]

        scaling = 1./max(1.,t+start_t)
        new_W_delta =  -scaling * W
        # check the loss
        if (loss=='hinge' and true_class_best_score - not_class_best_score < 1) or (loss=='ramp' and abs(true_class_best_score - not_class_best_score) < 1):
            eta = scaling/l
            new_W_delta[true_class_best_score_id] += eta*X[example_id]
            new_W_delta[not_class_best_score_id] -= eta*X[example_id]

        if return_avg_W:
            avg_W_update += (new_W_delta - avg_W_update)/(t+1)

        W += new_W_delta



        if do_projection:
            if type(do_projection) == int:
                if t % do_projection != 0: continue
            W_norm = np.linalg.norm(W)
            if W_norm == 0:
                return_tuple = (W,)
                if return_avg_W:
                    return_tuple += (init_W + avg_W_update,)
                if return_loss:
                    return_tuple += (loss_list,)
                if return_avg_W and return_loss:
                    return_tuple += (avg_W_loss_list,)
                return return_tuple

            scaling = oneoversqrtlambda/W_norm
            W *= min(1,scaling)

        # compute the zero-one loss to check for convergence
        if loss_computation > 0 and (t % loss_computation == 0):
            loss_list.append((t,(np.dot(X,W.T).argmax(1) == Y).sum() / X.shape[0]))
            print "round %d: loss=%g" % (t,loss_list[-1][-1] )

            if return_avg_W:
                avg_W_loss_list.append((t, (np.dot(X,(init_W+avg_W_update).T).argmax(1) == Y).sum() / X.shape[0]))
                print "round %d: loss avgW=%g" % (t, avg_W_loss_list[-1][-1])

    return_tuple = (W,)
    if return_avg_W:
        return_tuple += (init_W + avg_W_update,)
    if return_loss:
        return_tuple += (loss_list,)
    if return_avg_W and return_loss:
        return_tuple += (avg_W_loss_list,)
    return return_tuple


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




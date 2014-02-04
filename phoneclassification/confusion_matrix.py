from __future__ import division
import numpy as np

def confusion_matrix(guesses,labels):
    """
    Given a list of guesses and the true labels compute the confusion matrix
    Assumption is that the labels start at 0 and end at the highest index
    output is a matrix C where
    C[i,j] is the number of times a datum with true label i was predicted to be
    label j
    The set of i labels is [0,...,labels.max()] = np.arange(labels.max()+1)
    The set of j guesses is [0,...,max(guesses.max(),labels.max()) ]= np.arange(max(guesses.max(),labels.max())+1)
    """
    L = labels.max() + 1
    G = max(L,guesses.max()+1)
    C = np.zeros((L,G),dtype=int)
    for l in xrange(L):
        I_l = (labels == l)
        for g in xrange(G):
            C[l,g] = ((guesses == g) * I_l).sum()

    return C

def normalize_confusion_rows(C):
    return C / C.sum(1)[:,np.newaxis]
    

#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int_t INT
ctypedef np.uint8_t BINARY
ctypedef np.float32_t FLOAT

from libc.math cimport fabs

cdef extern from "math.h":
    float logf(float x)
    float expf(float x)
    float fmaxf(float x, float y)


def confusion(np.ndarray[INT,ndim=1] Lhat,
              np.ndarray[INT,ndim=1] L,
              INT n_labels):
    """Compute the confusion matrix using estimated labels
    
    Parameters
    ----------
    Lhat : array, shape = [n_samples,], dtype = np.int
        Estimated component labels
    
    L : array, shape = [n_samples,], dtype = np.int
        Ground truth labels

    n_labels : int
        number of labels

    Returns
    -------
    C : array-like, shape = [n_labels, n_labels]
    """
    cdef int n_samples = Lhat.shape[0]
    cdef np.ndarray[INT,ndim=2] C = np.zeros((n_labels,n_labels),dtype=np.int)
    cdef int i

    for i in range(n_samples):
        C[L[i],Lhat[i]] += 1

    return C


def labellogsumexp(np.ndarray[FLOAT,ndim=2] lpr,
                   np.ndarray[INT,ndim=1] L,
                   np.ndarray[BINARY,ndim=2] labels_to_components,
                   np.ndarray[FLOAT,ndim=1] labellogprob
                   ):
    """
    """
    cdef int n_samples = L.shape[0]
    cdef int n_labels = labels_to_components.shape[0]
    cdef int n_components = lpr.shape[1]
    cdef int sample_id, component_id, set_max
    cdef float lpr_max


    for sample_id in range(n_samples):
        set_max = 0
        labellogprob[sample_id] = 0
        for component_id in range(n_components):
            if labels_to_components[L[sample_id],component_id] > 0 :
                if set_max < 1:
                    lpr_max = lpr[sample_id,component_id]
                    set_max = 1
                else:
                    lpr_max = fmaxf(lpr[sample_id,component_id],
                                    lpr_max)

        for component_id in range(n_components):
            if labels_to_components[L[sample_id],component_id] > 0:

                labellogprob[sample_id] += expf(
                    lpr[sample_id,component_id] - lpr_max)

        labellogprob[sample_id] = logf(labellogprob[sample_id]) + lpr_max
    
            
def labelresponsibilities(np.ndarray[FLOAT,ndim=2] lpr,
                          np.ndarray[FLOAT,ndim=1] labellogprob,
                          np.ndarray[INT,ndim=1] L,
                          np.ndarray[BINARY,ndim=2] labels_to_components,
                          np.ndarray[FLOAT,ndim=2] responsibilities):
    """
    """
    cdef int n_samples = L.shape[0]
    cdef int n_labels = labels_to_components.shape[0]
    cdef int n_components = lpr.shape[1]
    cdef int sample_id, component_id, set_max
    cdef float lpr_max


    for sample_id in range(n_samples):
        for component_id in range(n_components):
            responsibilities[sample_id,component_id] = 0
            if labels_to_components[L[sample_id],component_id] > 0 :
                responsibilities[sample_id,component_id] = expf(lpr[sample_id,component_id] - labellogprob[sample_id])

    
            

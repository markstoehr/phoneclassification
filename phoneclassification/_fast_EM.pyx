# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mark Stoehr
#
# Based on code by:
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel (partial_fit support)
#         Rob Zinkov (passive-aggressive)
#         Lars Buitinck
#         
#
# Licence: BSD 3 clause


import numpy as np
import sys
from time import time

cimport cython
from libc.limits cimport INT_MAX
from libc.math cimport exp, log, sqrt, pow, fabs
cimport numpy as np

cdef inline double double_min(double a, double b) nogil: return a if a <= b else b
cdef inline double double_max(double a, double b) nogil: return b if a <= b else a
cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int, double *, int, double *, int) nogil
    void dscal "cblas_dscal"(int, double, double *, int) nogil


np.import_array()

# Learning rate constants
DEF CONSTANT = 1
DEF OPTIMAL = 2
DEF INVSCALING = 3
DEF PA1 = 4
DEF PA2 = 5

DEF EPSILON = 0.000001
DEF LOGEPSILON = -13.8156

# setup bool types
BOOL = np.uint8
ctypedef np.uint8_t BOOL_t
SHORT = np.int16
ctypedef np.int16_t SHORT_t


cdef void _sparse_dot(int *X_indices_ptr, int *rownnz,
                      int* rowstartidx, int X_n_rows, double *w_ptr, double *z_ptr) nogil:
    """
    """
    cdef int i,j, idx
    cdef int *cur_X_ptr = <int *>X_indices_ptr
    for i in range(X_n_rows):
        z_ptr[i] = 0.0
        for j in range(rownnz[i]):
            idx = cur_X_ptr[j]
            z_ptr[i] += w_ptr[idx]

        cur_X_ptr += rownnz[i]

cdef void _col_sum(double *A,double *A_col_sums,int X_n_rows, int K) nogil:
    cdef int i,j
    cdef double *cur_A_ptr = <double *>A
    
    for j in range(K):
        A_col_sums[j] = 0.0

    for i in range(X_n_rows):
        for j in range(K):
            A_col_sums[j] += cur_A_ptr[j]

        cur_A_ptr += K

cdef void _weighted_col_sum(double *A,double * u,double *A_col_sums,int X_n_rows, int K) nogil:
    cdef: 
        int i,j
        double *cur_A_ptr = <double *>A
        double *u_ptr = <double *> u
    
    
    for j in range(K):
        A_col_sums[j] = 0.0

    for i in range(X_n_rows):
        for j in range(K):
            A_col_sums[j] += u_ptr[0] * cur_A_ptr[j]

        if i < X_n_rows-1:
            cur_A_ptr += K
            u_ptr += 1
        

def col_sum(np.ndarray[double,ndim=1,mode="c"] A,

            int X_n_rows,
            int K):
    cdef np.ndarray[double,ndim=1,mode="c"] A_col_sums = np.zeros(K,dtype=np.float)
    _col_sum(<double *>A.data,<double *>A_col_sums.data,X_n_rows,K)
    return A_col_sums


def m_step(np.ndarray[int,ndim=1,mode="c"] X_indices, 
           np.ndarray[int,ndim=1,mode="c"] rownnz, 
           np.ndarray[int,ndim=1,mode="c"] rowstartidx,
           np.ndarray[double,ndim=1,mode="c"] P, 
           np.ndarray[double,ndim=1,mode="c"] weights, 
           np.ndarray[double,ndim=1,mode="c"] A, int X_n_rows, int D, int K):
    cdef: 
        np.ndarray[double, ndim=1, mode='c'] A_col_sums = np.zeros(K,dtype=np.float)

    _m_step(<int* >X_indices.data, <int *>rownnz.data, <int *> rowstartidx.data,
                 <double *>P.data, 
            <double *>weights.data, <double *>A.data, <double *>A_col_sums.data,X_n_rows, D, K)


cdef void _m_step(int* X_indices, int*rownnz, int * rowstartidx,
                 double *P, double *weights, double *A, double *A_col_sums, int X_n_rows, int D, int K) nogil:
    cdef int k,i,j,idx
    cdef int *cur_X_ptr = <int *> X_indices
    cdef double *cur_P_ptr = <double *> P
    cdef double *cur_A_ptr = <double *> A
    cdef double total_A_sum = 0.0
    _col_sum(A,A_col_sums,X_n_rows, K)
    cur_P_ptr = <double *> P

    for k in range(K):
        for j in range(D):
            cur_P_ptr[j] = 0.0
            
        cur_A_ptr = <double *> A
        total_A_sum += A_col_sums[k]
        cur_X_ptr = <int *> X_indices

        for i in range(X_n_rows):            
            for j in range(rownnz[i]):
                idx = cur_X_ptr[j]
                cur_P_ptr[idx] += cur_A_ptr[k]

            cur_A_ptr += K
            cur_X_ptr += rownnz[i]
        cur_P_ptr += D


    cur_P_ptr = <double *> P
    for k in range(K):
        weights[k] = A_col_sums[k]/total_A_sum
        for j in range(D):
            cur_P_ptr[0] /= A_col_sums[k]
            cur_P_ptr[0] = double_min(double_max(cur_P_ptr[0],.01),.99)
            cur_P_ptr += 1
    

cdef void _weighted_m_step(int* X_indices, int*rownnz, double * u,
                 double *P, double *weights, double *A, double *A_col_sums, int X_n_rows, int D, int K) nogil:
    """
    u are the weights for each datum
    """
    cdef int k,i,j,idx
    cdef int *cur_X_ptr = <int *> X_indices
    cdef double *cur_P_ptr = <double *> P
    cdef double *cur_A_ptr = <double *> A
    cdef double total_A_sum = 0.0
    cdef double *u_ptr = <double *> u
    _weighted_col_sum(A,u,A_col_sums,X_n_rows, K)
    cur_P_ptr = <double *> P

    for k in range(K):
        for j in range(D):
            cur_P_ptr[j] = 0.0
            
        cur_A_ptr = <double *> A
        total_A_sum += A_col_sums[k]
        cur_X_ptr = <int *> X_indices

        u_ptr = <double *> u
        for i in range(X_n_rows):            
            for j in range(rownnz[i]):
                idx = cur_X_ptr[j]
                cur_P_ptr[idx] += u_ptr[0] * cur_A_ptr[k]

            u_ptr += 1
            cur_A_ptr += K
            cur_X_ptr += rownnz[i]
        cur_P_ptr += D


    cur_P_ptr = <double *> P
    for k in range(K):
        weights[k] = A_col_sums[k]/total_A_sum
        for j in range(D):
            cur_P_ptr[0] /= A_col_sums[k]
            cur_P_ptr[0] = double_min(double_max(cur_P_ptr[0],.01),.99)
            cur_P_ptr += 1
    


cdef double _e_step(int* X_indices, int*rownnz,
                    double *P, double *log_inv_P_sum, double *weights, double *A, int X_n_rows, int D, int K,int compute_logodds) nogil:
    cdef: 
        int i,j,k,idx
        double *P_ptr = <double *> P
        double *log_inv_P_sum_ptr = <double *> log_inv_P_sum
        double log_inv, loglikelihood, maxloglike, prob_sum
        int *cur_X_ptr = <int *> X_indices
        int *X_ptr
        double *A_ptr
        int maxloglike_idx

    if compute_logodds > 0:
        for k in range(K):
            weights[k] = log(weights[k])
            log_inv_P_sum_ptr[k] = 0.0
            for j in range(D):
                log_inv = log(1.0-P_ptr[0])
                log_inv_P_sum_ptr[k] += log_inv
                P_ptr[0] = log(P_ptr[0]) - log_inv
                P_ptr += 1



    log_inv_P_sum_ptr = <double *> log_inv_P_sum
    P_ptr = <double *>P
    loglikelihood = 0.0
    for k in range(K):
        A_ptr = <double *> A
        X_ptr = <int *> X_indices
        for i in range(X_n_rows):
            A_ptr[k] = weights[k] + log_inv_P_sum_ptr[k]
            for j in range(rownnz[i]):
                idx = X_ptr[j]
                A_ptr[k] += P_ptr[idx]
                
            
            A_ptr += K
            X_ptr += rownnz[i]

        P_ptr += D

    X_ptr = <int *> X_indices
    A_ptr = <double *> A
    for i in range(X_n_rows):
        maxloglike = A_ptr[0]
        maxloglike_idx = 0
        prob_sum = 0.0
        for k in range(K):
            if A_ptr[k] > maxloglike:
                maxloglike_idx = k
                maxloglike = A_ptr[maxloglike_idx]
                
        for k in range(K):
            A_ptr[k] = exp(A_ptr[k] - maxloglike)
            prob_sum += A_ptr[k]
            
        loglikelihood += log(prob_sum) + maxloglike
        
        for k in range(K):
            A_ptr[k]/= prob_sum

        A_ptr += K
            
    return loglikelihood

def e_step(np.ndarray[int,ndim=1,mode="c"] X_indices, 
           np.ndarray[int,ndim=1,mode="c"] rownnz, 
           np.ndarray[int,ndim=1,mode="c"] rowstartidx,
           np.ndarray[double,ndim=1,mode="c"] P, 
           np.ndarray[double,ndim=1,mode="c"] weights, 
           np.ndarray[double,ndim=1,mode="c"] A, int X_n_rows, int D, int K):
    cdef: 
        np.ndarray[double,ndim=1,mode="c"] log_inv_P_sum = np.zeros(K,dtype=np.float)

    return _e_step(<int* >X_indices.data, <int *>rownnz.data,
                 <double *>P.data, <double *>log_inv_P_sum.data, 
                   <double *>weights.data, <double *>A.data, X_n_rows, D, K,1)
    
cdef void logtable(double *v, int N) nogil:
    cdef int i
    for i in range(N):
        v[i] = log(v[i])


def EM(np.ndarray[int,ndim=1,mode='c'] X_indices,
       np.ndarray[int,ndim=1, mode='c'] rownnz,
       np.ndarray[int, ndim=1, mode='c'] rowstartidx,
       int X_n_rows,
       int D, int K,
       double tol, int total_iter,
       np.ndarray[double, ndim=1, mode='c'] A):
    """
    P probability models  K by D
    A assignments X_n_rows by K -- assumed to be initialized randomly
    """
    cdef: 
        np.ndarray[double, ndim=1, mode='c'] P = np.zeros(D*K,dtype=np.float)
        np.ndarray[double, ndim=1, mode='c'] weights = np.zeros(K,dtype=np.float)
        np.ndarray[double, ndim=1, mode='c'] log_inv_P_sum = np.zeros(K,dtype=np.float)
        np.ndarray[double, ndim=1, mode='c'] A_col_sums = np.zeros(K,dtype=np.float)
        double old_loglikelihood, loglikelihood
        int n_iter = 0

    _m_step(<int *>X_indices.data, <int *>rownnz.data, <int *>rowstartidx.data,
            <double *>P.data, <double *>weights.data, <double *>A.data, <double *>A_col_sums.data, X_n_rows, D, K)


    loglikelihood = _e_step(<int *>X_indices.data, <int *>rownnz.data, 
                           
                            <double *>P.data, <double *>log_inv_P_sum.data, <double *>weights.data, 
                            <double *>A.data,  X_n_rows, D, K,1)
    old_loglikelihood = (1.0+tol)* loglikelihood -1.0
    
    
    while (loglikelihood - old_loglikelihood >  -tol *old_loglikelihood ) and (n_iter < total_iter):
        _m_step(<int *>X_indices.data, <int *>rownnz.data, <int *>rowstartidx.data,
                 <double *>P.data, <double *>weights.data, <double *>A.data, <double *>A_col_sums.data, X_n_rows, D, K)
        old_loglikelihood = loglikelihood
        loglikelihood = _e_step(<int *>X_indices.data, <int *>rownnz.data, 
                            <double *>P.data, <double *>log_inv_P_sum.data, <double *>weights.data, 
                                <double *>A.data, X_n_rows, D, K,1)
        n_iter += 1
        print ("Iteration %d: loglikelihood = %g " % (n_iter,loglikelihood))

    _m_step(<int *>X_indices.data, <int *>rownnz.data, <int *>rowstartidx.data,
                 <double *>P.data, <double *>weights.data, <double *>A.data, <double *>A_col_sums.data, X_n_rows, D, K)
    print "Finished Last M-step"
    return P, weights, A, loglikelihood

def soft_m_step(np.ndarray[int,ndim=1,mode="c"] X_indices, 
                np.ndarray[np.uint8_t,ndim=1,mode="c"] X_values, 
           np.ndarray[int,ndim=1,mode="c"] rownnz, 
           np.ndarray[double,ndim=1,mode="c"] P, 
           np.ndarray[double,ndim=1,mode="c"] weights, 
           np.ndarray[double,ndim=1,mode="c"] A, double x_base, int X_n_rows, int D, int K):
    cdef: 
        np.ndarray[double, ndim=1, mode='c'] A_col_sums = np.zeros(K,dtype=np.float)

    _soft_m_step(<int* >X_indices.data, <np.uint8_t *> X_values.data, x_base, <int *>rownnz.data, 
                 <double *>P.data, 
            <double *>weights.data, <double *>A.data, <double *>A_col_sums.data,X_n_rows, D, K)


cdef void _soft_m_step(int* X_indices, np.uint8_t* X_values, double x_base_double, int*rownnz, 
                 double *P, double *weights, double *A, double *A_col_sums, int X_n_rows, int D, int K) nogil:
    cdef int k,i,j,idx
    cdef int *cur_X_ptr = <int *> X_indices
    cdef np.uint8_t *cur_X_val_ptr = <np.uint8_t *> X_values
    cdef double *cur_P_ptr = <double *> P
    cdef double *cur_A_ptr = <double *> A
    cdef double total_A_sum = 0.0
    _col_sum(A,A_col_sums,X_n_rows, K)
    cur_P_ptr = <double *> P



    for k in range(K):
        for j in range(D):
            cur_P_ptr[j] = 0.0
            
        cur_A_ptr = <double *> A
        total_A_sum += A_col_sums[k]
        cur_X_ptr = <int *> X_indices
        cur_X_val_ptr = <np.uint8_t *> X_values
        
        for i in range(X_n_rows):            
            # print "example_id = %d" % i
            for j in range(rownnz[i]):
                # print "nonzero index = %d" % j
                idx = cur_X_ptr[j]
                # print "got the idx = %d" % idx
                # print "value is = %g" %  (<double> cur_X_val_ptr[j])
                # print "current A value = %g" % cur_A_ptr[k]
                # print "current P value = %g" % cur_P_ptr[idx]

                cur_P_ptr[idx] += cur_A_ptr[k] * (<double> cur_X_val_ptr[j])

            cur_A_ptr += K
            cur_X_ptr += rownnz[i]
            cur_X_val_ptr += rownnz[i]
            
        for j in range(D):
            cur_P_ptr[j] /= x_base_double
            
        cur_P_ptr += D


    cur_P_ptr = <double *> P
    for k in range(K):
        weights[k] = A_col_sums[k]/total_A_sum
        for j in range(D):
            cur_P_ptr[0] /= A_col_sums[k]
            cur_P_ptr[0] = double_min(double_max(cur_P_ptr[0],.01),.99)
            cur_P_ptr += 1
    

def soft_e_step(np.ndarray[int,ndim=1,mode="c"] X_indices, 
                np.ndarray[np.uint8_t,ndim=1,mode="c"] X_values, 
           np.ndarray[int,ndim=1,mode="c"] rownnz, 
           np.ndarray[double,ndim=1,mode="c"] P, 
           np.ndarray[double,ndim=1,mode="c"] weights, 
           np.ndarray[double,ndim=1,mode="c"] A, double x_base,
                int X_n_rows, int D, int K):
    cdef: 
        np.ndarray[double,ndim=1,mode="c"] log_inv_P_sum = np.zeros(K,dtype=np.float)

    return _soft_e_step(<int* >X_indices.data, <np.uint8_t *> X_values.data, x_base, <int *>rownnz.data,
                 <double *>P.data, <double *>log_inv_P_sum.data, 
                   <double *>weights.data, <double *>A.data, X_n_rows, D, K,1)


cdef double _soft_e_step(int* X_indices, np.uint8_t* X_values, double x_base, int*rownnz,
                    double *P, double *log_inv_P_sum, double *weights, double *A, int X_n_rows, int D, int K,int compute_logodds) nogil:
    cdef: 
        int i,j,k,idx
        double *P_ptr = <double *> P
        double *log_inv_P_sum_ptr = <double *> log_inv_P_sum
        double log_inv, loglikelihood, maxloglike, prob_sum
        int *cur_X_ptr = <int *> X_indices
        int *X_ptr
        np.uint8_t *X_val_ptr
        double *A_ptr
        int maxloglike_idx

    if compute_logodds > 0:
        for k in range(K):
            weights[k] = log(weights[k])
            log_inv_P_sum_ptr[k] = 0.0
            for j in range(D):
                log_inv = log(1.0-P_ptr[0])
                log_inv_P_sum_ptr[k] += log_inv
                P_ptr[0] = log(P_ptr[0]) - log_inv
                P_ptr += 1



    log_inv_P_sum_ptr = <double *> log_inv_P_sum
    P_ptr = <double *>P
    loglikelihood = 0.0
    for k in range(K):
        A_ptr = <double *> A
        X_ptr = <int *> X_indices
        X_val_ptr = <np.uint8_t *> X_values
        for i in range(X_n_rows):
            A_ptr[k] = weights[k] + log_inv_P_sum_ptr[k]
            for j in range(rownnz[i]):
                idx = X_ptr[j]
                A_ptr[k] += P_ptr[idx] *(<double> X_val_ptr[j])/x_base
                
            
            A_ptr += K
            X_ptr += rownnz[i]
            X_val_ptr += rownnz[i]

        P_ptr += D

    A_ptr = <double *> A
    for i in range(X_n_rows):
        maxloglike = A_ptr[0]
        maxloglike_idx = 0
        prob_sum = 0.0
        for k in range(K):
            if A_ptr[k] > maxloglike:
                maxloglike_idx = k
                maxloglike = A_ptr[maxloglike_idx]
                
        for k in range(K):
            A_ptr[k] = exp(A_ptr[k] - maxloglike)
            prob_sum += A_ptr[k]
            
        loglikelihood += log(prob_sum) + maxloglike
        
        for k in range(K):
            A_ptr[k]/= prob_sum

        A_ptr += K
            
    return loglikelihood



def soft_bernoulli_EM(np.ndarray[int,ndim=1,mode='c'] X_indices,
                      np.ndarray[np.uint8_t,ndim=1,mode='c'] X_values,
                      double x_base,
       np.ndarray[int,ndim=1, mode='c'] rownnz,
       int X_n_rows,
       int D, int K,
       double tol, int total_iter,
       np.ndarray[double, ndim=1, mode='c'] A):
    """
    Handle the possibility of feature values between 0 and 1 as given
    by the X_values array in particular the value of the feature
    detected at X_indices[k] is X_values[k]/X_value_base \in (0,1)

    P probability models  K by D
    A assignments X_n_rows by K -- assumed to be initialized randomly
    """
    cdef: 
        np.ndarray[double, ndim=1, mode='c'] P = np.zeros(D*K,dtype=np.float)
        np.ndarray[double, ndim=1, mode='c'] weights = np.zeros(K,dtype=np.float)
        np.ndarray[double, ndim=1, mode='c'] log_inv_P_sum = np.zeros(K,dtype=np.float)
        np.ndarray[double, ndim=1, mode='c'] A_col_sums = np.zeros(K,dtype=np.float)
        double old_loglikelihood, loglikelihood
        int n_iter = 0

    
    _soft_m_step(<int *>X_indices.data, <np.uint8_t *>X_values.data, x_base, <int *>rownnz.data, 
            <double *>P.data, <double *>weights.data, <double *>A.data, <double *>A_col_sums.data, X_n_rows, D, K)


    loglikelihood = _soft_e_step(<int *>X_indices.data, <np.uint8_t *>X_values.data,x_base, <int *>rownnz.data, 
                           
                            <double *>P.data, <double *>log_inv_P_sum.data, <double *>weights.data, 
                            <double *>A.data,  X_n_rows, D, K,1)
    old_loglikelihood = (1.0+tol)* loglikelihood -1.0
    
    
    while (loglikelihood - old_loglikelihood >  -tol *old_loglikelihood ) and (n_iter < total_iter):
        _soft_m_step(<int *>X_indices.data, <np.uint8_t *>X_values.data, x_base, <int *>rownnz.data, 
                 <double *>P.data, <double *>weights.data, <double *>A.data, <double *>A_col_sums.data, X_n_rows, D, K)
        old_loglikelihood = loglikelihood
        loglikelihood = _soft_e_step(<int *>X_indices.data, <np.uint8_t *>X_values.data, x_base, <int *>rownnz.data, 
                            <double *>P.data, <double *>log_inv_P_sum.data, <double *>weights.data, 
                                <double *>A.data, X_n_rows, D, K,1)
        n_iter += 1
        print ("Iteration %d: loglikelihood = %g " % (n_iter,loglikelihood))

    _soft_m_step(<int *>X_indices.data, <np.uint8_t *>X_values.data, x_base, <int *>rownnz.data, 
                 <double *>P.data, <double *>weights.data, <double *>A.data, <double *>A_col_sums.data, X_n_rows, D, K)
    print "Finished Last M-step"
    return P, weights, A, loglikelihood
                      


def uncertainty_EM(np.ndarray[int,ndim=1,mode='c'] X_indices,
                   np.ndarray[int,ndim=1, mode='c'] rownnz,
                   np.ndarray[int,ndim=1,mode='c'] class_n_indices,
                   np.ndarray[int, ndim=1, mode='c'] class_n_data,
                   np.ndarray[int, ndim=1, mode='c'] y,
                   int n_classes,
                   int X_n_rows, int max_n_components_class,
                   int D, int n_models, int max_n_data_class,
                   int total_iter,
                   np.ndarray[double, ndim=1, mode='c'] P,
                   np.ndarray[double, ndim=1, mode='c'] W,
                   
                   np.ndarray[int, ndim=1, mode='c'] class_n_components,

                   np.ndarray[int, ndim=1, mode='c'] meta_classes,
                   int n_subiter
                 ):
    """
    basic algorithm is to start with an inner product of the data
    and the classes
    """
    cdef:
        np.ndarray[double,ndim=1,mode="c"] z = np.zeros(X_n_rows*n_models,dtype=np.float)
        np.ndarray[double,ndim=1,mode="c"] u = np.ones(X_n_rows,dtype=np.float)
        np.ndarray[double,ndim=1,mode="c"] log_inv_P_sum = np.zeros(n_models,dtype=np.float)
        np.ndarray[double, ndim=1, mode='c'] A = np.zeros(max_n_data_class*max_n_components_class,dtype=np.float)
        np.ndarray[double, ndim=1, mode='c'] A_col_sums = np.zeros(max_n_components_class,dtype=np.float)
        
        double uncertainty, loglikelihood
        double * P_ptr = <double *> P.data
        double * log_inv_P_sum_ptr = <double *> log_inv_P_sum.data
        double * weights_ptr = <double *> W.data
        int * cur_X_ptr = <int *> X_indices.data
        int * rownnz_ptr = <int *> rownnz.data
        int iteration_id, c, subiter_id, compute_logodds

    for iteration_id in range(total_iter):
        _model_loglikes(<int *> X_indices.data, <int *> rownnz.data,
                    X_n_rows,<double *>P.data,
                    <double *> log_inv_P_sum.data,
                    <double *> W.data,
                        <double *>z.data, D, n_models,0)
        uncertainty = _uncertainty_weights(<double *>z.data,<int *> y.data,
                                           <int *> meta_classes.data,
                                       <double *> u.data, n_models, X_n_rows)


        print ("Iteration %d: uncertainty = %g" % (iteration_id,uncertainty))
        P_ptr = <double *> P.data
        log_inv_P_sum_ptr = <double *> log_inv_P_sum.data
        weights_ptr = <double *> W.data
        cur_X_ptr = <int *> X_indices.data
        rownnz_ptr = <int *> rownnz.data
        for c in range(n_classes):
            print ("class_n_data[%d] = %d\t class_n_components[%d]= %d" % (c,class_n_data[c],c,class_n_components[c]))
            compute_logodds = 0
            for subiter_id in range(n_subiter):
                loglikelihood = _e_step(cur_X_ptr, rownnz_ptr,
                    P_ptr, log_inv_P_sum_ptr, 
                    weights_ptr, <double *> A.data, 
                                        class_n_data[c], D, class_n_components[c],compute_logodds)
                print ("class %d: loglikelihood = %g" % (c,loglikelihood))

                _weighted_m_step(cur_X_ptr, rownnz_ptr,
                             <double *> u.data,
                             P_ptr, weights_ptr, <double *> A.data,
                             <double *> A_col_sums.data,
                             class_n_data[c],D,class_n_components[c])
                compute_logodds = 1
            if c < n_classes-1:
                P_ptr += D* class_n_components[c]
                log_inv_P_sum_ptr +=  class_n_components[c]
                weights_ptr += class_n_components[c]
                cur_X_ptr += class_n_indices[c]
                rownnz_ptr += class_n_data[c]

            

    return uncertainty, P, W

def model_loglikes(np.ndarray[ndim=1,dtype=int,mode="c"] X_indices,
                   np.ndarray[ndim=1,dtype=int,mode="c"] rownnz,
                   int X_n_rows,
                   np.ndarray[ndim=1,dtype=double,mode="c"] P,
                   np.ndarray[ndim=1,dtype=double,mode="c"] Weights,
                   int D, int n_models):
    cdef:
       np.ndarray[ndim=1,dtype=double,mode="c"] log_P_inv_sum = np.zeros(n_models,dtype=np.float)
       np.ndarray[ndim=1,dtype=double,mode="c"] z = np.zeros(n_models*X_n_rows,dtype=np.float)

    _model_loglikes(<int *>X_indices.data,
                    <int *> rownnz.data,
                    X_n_rows,
                    <double *> P.data,
                    <double *> log_P_inv_sum.data,
                    <double *> Weights.data,
                    <double *> z.data,
                    D,n_models,1)

    return z.reshape(X_n_rows,n_models)

cdef double _model_loglikes(int * X_indices, int * rownnz,
                    int X_n_rows,double *P,
                    double * log_inv_P_sum,
                    double * Weights,
                            double *z, int D, int n_models, int compute_log_odds) nogil:
    cdef:
        int i,j,k, idx
        double *P_ptr = <double *> P
        double *log_inv_P_sum_ptr = <double *> log_inv_P_sum
        double log_inv
        int *cur_X_ptr = <int *> X_indices
        double * z_ptr = <double *> z
        int * rownnz_ptr = <int *> rownnz

    for i in range(n_models):
        Weights[i] = log(Weights[i])
        log_inv_P_sum_ptr[i] = 0.0
        for j in range(D):
            log_inv = log(1.0 -P_ptr[0])
            log_inv_P_sum_ptr[i] += log_inv
            P_ptr[0] = log(P_ptr[0]) - log_inv
            P_ptr += 1

    log_inv_P_sum_ptr = <double *> log_inv_P_sum
    P_ptr = <double *> P
    loglikelihood = 0.0
    for i in range(n_models):
        z_ptr = <double *> z
        z_ptr += i
        rownnz_ptr = <int *> rownnz
        cur_X_ptr = <int *> X_indices
        for j in range(X_n_rows):
            z_ptr[0] = Weights[i] + log_inv_P_sum_ptr[i]
            for k in range(rownnz_ptr[0]):
                idx = cur_X_ptr[k]
                z_ptr[0] += P_ptr[idx]
                
                
            if j < X_n_rows -1:
                z_ptr += n_models
                cur_X_ptr += rownnz_ptr[0]
                rownnz_ptr += 1
                                
                
        if i < n_models -1:
            P_ptr += D
                

def uncertainty_weights(
        np.ndarray[ndim=1,dtype=double,mode="c"] z,
        np.ndarray[ndim=1,dtype=int,mode="c"] y,
        np.ndarray[ndim=1,dtype=int,mode="c"] meta_classes,
        int n_models, int n_data):
    cdef np.ndarray[ndim=1,dtype=double,mode="c"] u = np.zeros(n_data,dtype=np.float)
    
    total_u = _uncertainty_weights(<double *> z.data,
                         <int *> y.data,
                         <int *> meta_classes.data,
                         <double *> u.data,
                         n_models, n_data)

    return u

cdef double _uncertainty_weights(double * z,
                                 int * y,
                                 int * meta_classes,
                                 double * u, int n_models,
                                 int n_data) nogil:
    cdef:
        int i,j, best_idx
        double best_score, max_diff, normalization, off_model_top, cur_score, total_u
        double * z_ptr = <double *> z
        int * y_ptr = <int *> y
        double * u_ptr = <double *> u
        

    total_u = 0.0
    for i in range(n_data):
        best_idx = 0
        best_score = z_ptr[0]
        for j in range(n_models):
            if z_ptr[j] > best_score:
                best_idx = j
                best_score = z_ptr[j]

        normalization = 0.0
        off_model_top = 0.0
        for j in range(n_models):
            cur_score = z_ptr[j] - best_score
            if cur_score > LOGEPSILON:
                cur_score = exp(cur_score)
                normalization += cur_score
                if meta_classes[j] == y_ptr[0]:
                    off_model_top += cur_score
        
        u_ptr[0] = off_model_top/normalization
        total_u += u_ptr[0]
        if i < n_data -1:
            z_ptr += n_models
            y_ptr += 1
            u_ptr += 1
    
    return total_u
    

def sparse_dotmm(np.ndarray[int,ndim=1,mode='c'] X_indices,
               np.ndarray[int,ndim=1, mode='c'] rownnz,
               np.ndarray[int, ndim=1, mode='c'] rowstartidx,
               np.ndarray[double,ndim=1,mode='c'] w,
                 int X_n_rows,
                 int D,
                 int K):
    cdef np.ndarray[double,ndim=1,mode="c"] z = np.zeros(X_n_rows*K,dtype=np.float)
    _sparse_dotmm(<int *> X_indices.data, <int *> rownnz.data,
                <int *> rowstartidx.data, 
                X_n_rows,<double *>w.data,
                  <double *>z.data, D, K)
    return z.reshape(X_n_rows,K)

cdef void _sparse_dotmm(int *X_indices_ptr, int *rownnz,
                        int* rowstartidx, int X_n_rows, double *w_ptr, double *z_ptr, int D, int K) nogil:
    """
    """
    cdef int i,j, k, z_idx, idx
    cdef int *cur_X_ptr = <int *>X_indices_ptr
    cdef double *cur_w_ptr = <double *>w_ptr
    z_idx = 0
    for i in range(X_n_rows):
        cur_w_ptr = <double *>w_ptr
        for k in range(K):
            z_ptr[z_idx] = 0.0
            for j in range(rownnz[i]):
                idx = cur_X_ptr[j]
                z_ptr[z_idx] += cur_w_ptr[idx]

            z_idx += 1
            cur_w_ptr += D

        cur_X_ptr += rownnz[i]

def sparse_soft_dotmm(np.ndarray[int,ndim=1,mode='c'] feature_ids,
                      np.ndarray[np.uint8_t,ndim=1,mode='c'] feature_values,
                      np.ndarray[int,ndim=1, mode='c'] rownnz,
               np.ndarray[double,ndim=1,mode='c'] w,
                      double value_base,
                 int D,
                 int K):
    cdef: 
        int n_data = rownnz.shape[0]
        np.ndarray[double,ndim=1,mode="c"] z = np.zeros(n_data*K,dtype=np.float)
    _sparse_soft_dotmm(<int *> feature_ids.data,
                       <np.uint8_t *> feature_values.data,
                       <int *> rownnz.data,
                n_data,<double *>w.data,
                  <double *>z.data, value_base, D, K)
    return z.reshape(n_data,K)


cdef void _sparse_soft_dotmm(int *feature_ids,
                             np.uint8_t *feature_values,
                             int *rownnz, int n_data, double *w_ptr, double *z_ptr, double value_base, int D, int K) nogil:
    """
    """
    cdef int i,j, k, z_idx, idx
    cdef int *cur_X_ptr = <int *>feature_ids
    cdef np.uint8_t *cur_X_val_ptr = <np.uint8_t *>feature_values
    cdef double *cur_w_ptr = <double *>w_ptr
    z_idx = 0
    for i in range(n_data):
        cur_w_ptr = <double *>w_ptr
        for k in range(K):
            z_ptr[z_idx] = 0.0
            for j in range(rownnz[i]):
                idx = cur_X_ptr[j]
                z_ptr[z_idx] += cur_w_ptr[idx] * (<double> cur_X_val_ptr[j])

            z_ptr[z_idx] /= value_base
            z_idx += 1
            cur_w_ptr += D

        cur_X_ptr += rownnz[i]
        cur_X_val_ptr += rownnz[i]


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
    
    for i in range(X_n_rows):
        for j in range(K):
            A_col_sums[j] += cur_A_ptr[j]

        cur_A_ptr += K

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
    

cdef double _e_step(int* X_indices, int*rownnz, int * rowstartidx,
                 double *P, double *log_inv_P_sum, double *weights, double *A, int X_n_rows, int D, int K) nogil:
    cdef: 
        int i,j,k,idx
        double *P_ptr = <double *> P
        double *log_inv_P_sum_ptr = <double *> log_inv_P_sum
        double log_inv, loglikelihood, maxloglike, prob_sum
        int *cur_X_ptr = <int *> X_indices
        int maxloglike_idx

    for k in range(K):
        weights[k] = log(weights[k])
        log_inv_P_sum_ptr[0] = 0.0
        for j in range(D):
            log_inv = log(1.0-P_ptr[0])
            log_inv_P_sum_ptr[0] += log_inv
            P_ptr[0] = log(P_ptr[0]) - log_inv
            P_ptr += 1

        log_inv_P_sum_ptr += 1

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

    return _e_step(<int* >X_indices.data, <int *>rownnz.data, <int *> rowstartidx.data,
                 <double *>P.data, <double *>log_inv_P_sum.data, 
            <double *>weights.data, <double *>A.data, X_n_rows, D, K)
    
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

    loglikelihood = _e_step(<int *>X_indices.data, <int *>rownnz.data, 
                            <int *>rowstartidx.data,
                            <double *>P.data, <double *>log_inv_P_sum.data, <double *>weights.data, 
                            <double *>A.data,  X_n_rows, D, K)
    old_loglikelihood = (1-tol)* loglikelihood -1
    while (loglikelihood - old_loglikelihood > tol *loglikelihood) and (n_iter < total_iter):
        _m_step(<int *>X_indices.data, <int *>rownnz.data, <int *>rowstartidx.data,
                 <double *>P.data, <double *>weights.data, <double *>A.data, <double *>A_col_sums.data, X_n_rows, D, K)
        old_loglikelihood = loglikelihood
        loglikelihood = _e_step(<int *>X_indices.data, <int *>rownnz.data, <int *>rowstartidx.data,
                            <double *>P.data, <double *>log_inv_P_sum.data, <double *>weights.data, 
                            <double *>A.data, X_n_rows, D, K)
        n_iter += 1

    return P, weights, A

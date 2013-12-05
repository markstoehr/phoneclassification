#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE
ctypedef np.int_t INT

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline int int_rem(int a, int b): return a - b *(a/b) 

from libc.math cimport fabs


def reassign(np.ndarray[DOUBLE,ndim=2] RS,
             np.ndarray[DOUBLE,ndim=2] S,
             np.ndarray[DOUBLE,ndim=2] DT,
             np.ndarray[DOUBLE,ndim=2] DF,
             np.ndarray[DOUBLE,ndim=2] RT,
             np.ndarray[DOUBLE,ndim=2] RF,
             np.ndarray[INT,ndim=2] below_thresh,
             DOUBLE threshold):
    cdef int t_id, f_id,t_hat_int,f_hat_int, ntimes, nfreq
    cdef double t_hat, f_hat


    ntimes = S.shape[0]
    nfreq = S.shape[1]

    for t_id in range(ntimes):
        for f_id in range(nfreq):
            if fabs(S[t_id,f_id]) > threshold:
                t_hat = t_id + DT[t_id,f_id]
                f_hat = f_id + DF[t_id,f_id]
                t_hat_int = int_min(int_max(<int>(t_hat+.5),0),ntimes-1)
                f_hat_int = int_rem(int_rem(<int>(f_hat+.5),nfreq)+nfreq,nfreq)
                RS[t_hat_int,f_hat_int] += S[t_id,f_id]
                RT[t_id,f_id] = t_hat
                RF[t_id,f_id] = f_hat
                below_thresh[t_id,f_id] = 0
            else:
                RS[t_id,f_id] += S[t_id,f_id]
                below_thresh[t_id,f_id] = 1
                RT[t_id,f_id] = 0
                RF[t_id,f_id] = 0



def just_reassign(np.ndarray[DOUBLE,ndim=2] RS,
                  np.ndarray[DOUBLE,ndim=2] S,
                  np.ndarray[DOUBLE,ndim=2] RT,
                  np.ndarray[DOUBLE,ndim=2] RF,
                  DOUBLE threshold):
                  
    cdef int t_id, f_id,t_hat_int,f_hat_int, ntimes, nfreq

    ntimes = S.shape[0]
    nfreq = S.shape[1]

    for t_id in range(ntimes):
        for f_id in range(nfreq):
            if fabs(S[t_id,f_id]) > threshold:
                t_hat_int = int_min(int_max(<int>(RT[t_id,f_id]+.5),0),ntimes-1)
                f_hat_int = int_rem(int_rem(<int>(RF[t_id,f_id]+.5),nfreq)+nfreq,nfreq)
                RS[t_hat_int,f_hat_int] += S[t_id,f_id]
            else:
                RS[t_id,f_id] += S[t_id,f_id]


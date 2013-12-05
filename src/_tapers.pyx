#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE

cdef extern from "math.h":
    double sqrt(double x)
 
def hermitetapers(unsigned int nx,unsigned int order,
                  DOUBLE av_val,
                  np.ndarray[DOUBLE,ndim=2] recurr,
                  np.ndarray[DOUBLE,ndim=2] tapers,
                  np.ndarray[DOUBLE,ndim=2] deriv_taper,
                  np.ndarray[DOUBLE,ndim=1] times,
                  
              ):
    """
    Parameters:
    ===========
    nx:
       number of sample points
    order:
       number of hermite tapers to generate
    """
    cdef:
        unsigned int taper_id, time_id
        DOUBLE deriv_coeff
        
    for taper_id in range(order+1):
        deriv_coeff = sqrt(2.0 * taper_id)
        
        for time_id in range(nx):
            # start using recurrence relation for the third taper
            if taper_id >= 2:
                recurr[taper_id,time_id] = 2* times[time_id] * recurr[taper_id-1,time_id] - 2*(taper_id-1)*recurr[taper_id-2,time_id]
                
            tapers[taper_id,time_id] *= recurr[taper_id,time_id] 

            if taper_id > 0:
                deriv_taper[taper_id-1,time_id] = av_val * (times[time_id] * tapers[taper_id-1,time_id] - deriv_coeff*tapers[taper_id,time_id])    
    

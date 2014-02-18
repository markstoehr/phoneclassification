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

cdef inline double double_min(double a, double b): return a if a <= b else b
cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int, double *, int, double *, int) nogil
    void dscal "cblas_dscal"(int, double, double *, int) nogil

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil
    void* malloc(size_t size) nogil
    void free(void* ptr) nogil

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


cdef struct PairItem:
    int i
    double *h

cdef int _compare(const_void *a, const_void *b) nogil:
    cdef double v = (<PairItem *>a)[0].h[0] - (<PairItem *>b)[0].h[0]
    if v > 0 : return 1
    elif v < 0 : return -1
    return 0
        
# cdef void _line_search( double *w_ptr, double *p_ptr, double l_param,
#                   double *f_ptr, double *delta_f_ptr,
#                         PairItem * eta_pi,
#                         double * eta,
#                         int D, int N) nogil:
#     cdef:
#         int i,j
#         double h = 0.0

#     for i in range(D):
#         h += p_ptr[i] * p_ptr[i]

#     h *= l_param
#     j = 1

#     # eta_ptr stores the eta values which are given by the dual
#     # non-differentiable points plus an extra zero at the beginning
#     # eta_pi exists as a way to allow argsorting

#     eta_ptr[0] = 0.0
#     eta_pi[0].i = 0
#     eta_pi[0].h = <double *>eta_ptr
#     for i in range(N):
#         eta_ptr[i+1] = (1.0-f_ptr[i])/delta_f_ptr[i]
#         eta_pi[i+1].i = i+1
#         eta_pi[i+1].h = <double *>(eta_ptr + (i+1)*sizeof(double))

#     qsort(eta_pi,N+1,sizeof(PairItem ),_compare)

cdef int _error_margin_subgradient(double * z_ptr, 
                                   int * error_counts,
                                   int * margin_indices,
                                   double * new_w,
                                   int D,
                                   int * X_indices,
                                   int * rownnz,
                                   int * rowstartidx,
                                   int X_n_rows) nogil:
    """
    Computes the subgradient loops through the data points
    and sees whether a given point is in error or on the margin
    """
    cdef: 
        int i,j,idx
        int margin_idx = 0
        int cur_beta_idx = 0
        double inverse_n_points = 1./(<double>X_n_rows)

    cdef int *cur_X_ptr = <int *>X_indices
    for i in range(X_n_rows):
        # if point is in error then add it to error set
        if z_ptr[i] > EPSILON -1:
            for j in range(rownnz[i]):
                idx = cur_X_ptr[j]
                error_counts[idx] += 1
        elif z_ptr[i] > -EPSILON-1:
            # margin point
            margin_indices[margin_idx] = i
            margin_idx += 1


    for i in range(D):
        new_w[i] += inverse_n_points * error_counts[i]
        error_counts[i] = 0
            

    return margin_idx

cdef double _exemplar_subgradient(
    double * new_w,
    double * g,
    int D,
    double * p,
    int * X_indices_ptr,
    int * rownnz,
    int *rowstartidx,
    int X_n_rows,
    int * margin_indices,
    int margin_nnz,
    double * z) nogil:
    """
    g is the gradient we are going to store
    we ouput the supremum over all subgradients
    for the current direction p
    """
    _sparse_dot_row_subset(X_indices_ptr,
                           rownnz, rowstartidx,
                           p, z,
                           margin_indices,
                           margin_nnz)
    cdef: 
        int i,j, row_idx, idx
        double one_over_n_rows = 1./(<double>X_n_rows)
        double subgradient = 0.0
        int *cur_X_ptr

    for j in range(D):
        g[j] = 0.0

    if margin_nnz > 0:
        for i in range(margin_nnz):
            if z[i] >0:
                row_idx = margin_indices[i]
                cur_X_ptr = <int *>(& X_indices_ptr[rowstartidx[row_idx]])
                for j in range(rownnz[row_idx]):
                    idx = cur_X_ptr[j]
                    g[idx] += 1.0
            
    


    for i in range(D):
        g[i] *= one_over_n_rows
        g[i] += new_w[i]
        subgradient += g[i] * p[i]

    return subgradient
        
cdef _LBFGS_two_loop_dot(double * rhos,
                         double * ys,
                         double * ss,
                         int D,
                         int * entry_indices,

def find_descent_direction(np.ndarray[int,ndim=1,mode='c'] X_indices,
                           np.ndarray[int,ndim=1, mode='c'] rownnz,
                           np.ndarray[int, ndim=1, mode='c'] rowstartidx,
                           np.ndarray[double,ndim=1,mode='c'] w,
                           int X_n_rows):
    """
    """
    cdef: 
        np.ndarray[double, ndim=1,mode="c"] z = np.zeros(X_n_rows,dtype=np.float)
        int i, margin_nnz
        int D = w.shape[0]
        np.ndarray[ndim=1,dtype=int,mode="c"] error_counts = np.zeros(D,dtype=np.intc)
        np.ndarray[ndim=1,dtype=int,mode="c"] margin_indices = np.zeros(X_n_rows,dtype=np.intc)

        np.ndarray[ndim=1,dtype=double,mode="c"] new_w = np.zeros(D,dtype=np.float)
        np.ndarray[ndim=1,dtype=double,mode="c"] subgradient = np.zeros(D,dtype=np.float)
        np.ndarray[ndim=1,dtype=double,mode="c"] new_subgradient = np.zeros(D,dtype=np.float)
        np.ndarray[ndim=1,dtype=double,mode="c"] descent_vec = np.zeros(D,dtype=np.float)
        np.ndarray[ndim=1,dtype=double,mode="c"] new_descent_vec = np.zeros(D,dtype=np.float)
        np.ndarray[ndim=1,dtype=double,mode="c"] best_descent_vec = np.zeros(D,dtype=np.float)
        
        # get the constants

    # get the initial score
    _sparse_dot(<int *> X_indices.data, <int *> rownnz.data,
                <int *> rowstartidx.data, 
                X_n_rows,<double *>w.data,
                <double *>z.data)

    for i in range(D):
        new_w[i] = w[i] * l_param

    margin_nnz = _error_margin_subgradient(<double *>z.data, <int *> error_counts.data, <int *> margin_indices.data, <double *> new_w.data, D, <int *> X_indices.data, <int *> rownnz.data, <int *> rowstartidx.data,
                                                  X_n_rows)
    
    _base_subdifferential



def compare(np.ndarray[ndim=1,dtype=double,mode="c"] eta):
    cdef: 
        Py_ssize_t i 
        int c
        void * eta_ptr = <const_void *>eta.data
        int n = eta.shape[0]
        PairItem *eta_pi = <PairItem *>malloc(n * sizeof(PairItem))

    for i in range(n):
        eta_pi[i].i = i;
        eta_pi[i].h = <double *>(eta.data + i*sizeof(double))

    for i in range(n):
        print ("eta[%d] = %g" % (eta_pi[i].i, (eta_pi[i]).h[0]))
    
    qsort(eta_pi,n,sizeof(PairItem ),_compare)
    for i in range(n):
        print ("eta[%d] = %g" % (eta_pi[i].i, (eta_pi[i]).h[0]))

    # for i in range(n):
    #     c = _compare(eta_ptr, eta_ptr + sizeof(double))
    #     eta_ptr += sizeof(double)
    #     if c == 1:
    #         print ("eta[%d] = %g > eta[%d] = %g" % (i-1,eta[i-1],i,eta[i]))
    #     elif c == -1:
    #         print ("eta[%d] = %g < eta[%d] = %g" % (i-1,eta[i-1],i,eta[i]))
    #     else:
    #         print ("eta[%d] = %g == eta[%d] = %g" % (i-1,eta[i-1],i,eta[i]))

    


cdef class MultiWeightMatrix(object):
    """Dense matrix represented by a 1d numpy array, a 2d numpy array,
    a scalar,
    and an index array.

    The class provides methods to ``add` a sparse vector
    to a row and scale the matrix.
    Representing a matrix explicitly as a scalar
    times a matrix allows for efficient scaling operations.

    Attributes
    ----------
    W : ndarray, dtype=double, order='C'
        The numpy array which backs the weight vector
    W_data_ptr : double*
        A pointer to the data of the numpy array.
    row_sq_norms : ndarray, dtype=double, order = 'C'
        A 1-d numpy array containing the squared norm of each
        row
    row_sq_norms_data_ptr : double *
        A pointer to the data in the ``row_sq_norms`` array
    scores : ndarray, dtype=double, order = 'C'
        A 1-d numpy array containing the response to multiplying
        ``W`` with a data vector, which can be used for
        computing losses
    scores_data_ptr : double*
        A pointer to the data in ``scores`` array
    wcale : double
        The scale of the matrix
    n_features : int
        The number of features (= row length of ``W``)
    n_components : int
        The number of rows (components) in ``W``
    I : ndarray, dtype=int, order='C'
        The numpy array which indexes the class and component
        structure, has the same number of rows as ``n_components``
        and two columns where the first column indicates the
        class associated with the row and the second column
        is the component id within that class
    sq_norm : double
        The squared norm of ``w``.
    """
    
    cdef Py_ssize_t n_components, n_entries
    cdef int n_features, n_classes
    cdef np.ndarray scores
    cdef double * scores_data_ptr
    cdef np.ndarray W
    cdef double * W_data_ptr
    cdef double Wscale, sq_norm
    cdef np.ndarray W_classes
    cdef int * W_classes_ptr
    cdef np.ndarray W_components
    cdef int * W_components_ptr
    cdef np.ndarray class_n_components
    cdef int * class_n_components_ptr
    cdef np.ndarray class_row_idx
    cdef int * class_row_idx_ptr
    
    # keep track of the averaged gradient
    

    # smoothed loss gradient and assignment parameters
    cdef double gamma, beta

    # components for computing the gradient
    cdef int best_true_row, best_off_row, n_use_off_scores, n_use_true_scores
    
    cdef double best_true_score, best_off_score, add_scaling, off_score_normalization, true_score_normalization

    cdef np.ndarray gradient
    cdef double * gradient_ptr
    
    cdef np.ndarray use_off_score_ids
    cdef int * use_off_score_ids_ptr
    cdef np.ndarray use_off_scores
    cdef double * use_off_scores_ptr
    cdef np.ndarray use_true_score_ids
    cdef int * use_true_score_ids_ptr
    cdef np.ndarray use_true_scores
    cdef double * use_true_scores_ptr

    def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] W,
                  np.ndarray[int, ndim=1, mode='c'] W_classes,
                  np.ndarray[int, ndim=1, mode='c'] W_components,
                  int n_classes, double gamma, double beta):
        """
        W should be put out as a single column vector

        Parameters :
        -------------
        gamma: double
            Parameter for computing the smoothed hinge loss gradient
        beta: double
            Parameter for computed the smoothed assignment vector
        """
        cdef double *wdata = <double *>W.data

        if W.shape[0] > INT_MAX or W.shape[1] > INT_MAX:
            raise ValueError("More than %d classes or features not supported; got (%d,%d)" % (INT_MAX, W.shape[0],W.shape[1]))
            


        self.n_components = W_classes.shape[0]
        self.n_classes = n_classes
        self.n_features = <int>(W.shape[0]/W_classes.shape[0])
        self.n_entries = <int>W.shape[0]

        cdef np.ndarray[double, ndim=1,
                        mode='c'] scores = np.zeros(self.n_components,
                                                    dtype= np.float)

        self.scores = scores
        self.scores_data_ptr = <double *>scores.data
        
        self.W = W
        self.W_data_ptr = wdata
        self.Wscale = 1.0
        
        self.W_classes = W_classes
        self.W_classes_ptr = <int *> W_classes.data
        
        self.W_components = W_components
        self.W_components_ptr = <int *> W_components.data

        cdef np.ndarray[int, ndim=1,
                                mode='c'] class_n_components = np.zeros(self.n_classes,
                                                                        dtype= np.intc)
        
        self.class_n_components = class_n_components
        self.class_n_components_ptr = <int *> class_n_components.data


        cdef np.ndarray[int, ndim=1,
                                mode='c'] class_row_idx = np.zeros(self.n_classes,
                                                                   dtype= np.intc)
        
        self.class_row_idx = class_row_idx
        self.class_row_idx_ptr = <int *> class_row_idx.data

        cdef int i, nrow_components
        nrow_components = 0
        for i in range(self.n_components):
            if W_components[i] == 0:
                if i > 0:
                    self.class_n_components_ptr[W_classes[i-1]] = nrow_components
                nrow_components = 1
                self.class_row_idx_ptr[W_classes[i]] = i
                
            else:
                nrow_components += 1

        self.class_n_components_ptr[W_classes[self.n_components-1]] = nrow_components

        self.beta = beta
        self.gamma = gamma

        self.sq_norm = ddot(self.n_entries,
                                             <double *>W.data, 1, <double *>W.data,1)
            
        # gradient
        cdef np.ndarray[double,ndim=1,mode='c'] gradient = np.zeros(
            self.n_entries,dtype=np.float)
        self.gradient = gradient
        self.gradient_ptr = <double *>gradient.data

        # relevant scores
        cdef np.ndarray[int,ndim=1,mode='c'] use_true_score_ids = np.zeros(
            self.n_components,dtype=np.intc)
        self.use_true_score_ids = use_true_score_ids
        self.use_true_score_ids_ptr = <int *>use_true_score_ids.data
    
        cdef np.ndarray[double,ndim=1,mode='c'] use_true_scores = np.zeros(
            self.n_components,dtype=np.float)
        self.use_true_scores = use_true_scores
        self.use_true_scores_ptr = <double *>use_true_scores.data

        cdef np.ndarray[int,ndim=1,mode='c'] use_off_score_ids = np.zeros(
            self.n_components,dtype=np.intc)
        self.use_off_score_ids = use_off_score_ids
        self.use_off_score_ids_ptr = <int *>use_off_score_ids.data
    
        cdef np.ndarray[double,ndim=1,mode='c'] use_off_scores = np.zeros(
            self.n_components,dtype=np.float)
        self.use_off_scores = use_off_scores
        self.use_off_scores_ptr = <double *>use_off_scores.data


    
    cdef void binary_add(self, int w_row,
                         int *x_ind_ptr, int xnnz, double c) nogil:
        """Scales binary vector sample x by constant c and adds it 
        to the weight vector.

        This operation updates ``sq_norm`` and ``row_sq_norms``
        Parameters
        ----------
        w_row : int
            The row of w we are using for classification
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``
        xnnz : int
            The number of non-zero features of ``x``.
        c : double
            The scaling constant for the example
        """
        cdef int j
        cdef int idx
        cdef int offset
        cdef double val
        cdef double innerprod = 0.0
        cdef double xsqnorm = 0.0
        
        cdef double Wscale = self.Wscale
        cdef double* W_data_ptr 



        
        W_data_ptr =  (self.W_data_ptr + w_row * self.n_features)
        for j in range(xnnz):
            idx = x_ind_ptr[j]
            innerprod += W_data_ptr[idx]
            xsqnorm += 1.0
            W_data_ptr[idx] += c / Wscale

        val = (xsqnorm * c * c) + (2.0 * innerprod * Wscale * c)
        self.sq_norm += val

    cdef void dot(self, int *x_ind_ptr,
                    int xnnz) nogil:
        """Computes the matrix-vector dot product between a sample x and the weight matrix. And stores the result in the weight vector.

        Parameters
        ----------
        x_ind_ptr : double*
            The array which holds the feature indices of ``x``
        xnnz : int
            The number of non-zero features of ``x`` (length of x_ind_ptr)


        """
        cdef int i,j
        cdef int idx
        cdef double innerprod = 0.0
        cdef double* scores_data_ptr = self.scores_data_ptr
        cdef double* W_data_ptr = self.W_data_ptr
        cdef double * cur_data_ptr

        for i in range(self.n_components):
            cur_data_ptr = W_data_ptr + i*self.n_features
            scores_data_ptr[i] = 0.0
            for j in range(xnnz):
                idx = x_ind_ptr[j]
                scores_data_ptr[i] += cur_data_ptr[idx]

    cdef void find_best_scores(self, int y) nogil:
        """Assumes scores have been computed with self.dot
        and finds the row of W that attained the highest score
        within class and outside the class
        """
        if y == 0:
            self.best_true_row = 0
            self.best_off_row = self.class_row_idx_ptr[1]
        else:
            self.best_true_row = self.class_row_idx_ptr[y]
            self.best_off_row = 0
            
        self.best_true_score = self.scores_data_ptr[self.best_true_row]
        self.best_off_score = self.scores_data_ptr[self.best_off_row]
        cdef int j
        for j in range(self.n_components):
            if self.W_classes_ptr[j] == y: # within class
                if self.scores_data_ptr[j] > self.best_true_score:
                    self.best_true_row = j
                    self.best_true_score = self.scores_data_ptr[j]
                    
            else: # outside class
                if self.scores_data_ptr[j] > self.best_off_score:
                    self.best_off_row = j
                    self.best_off_score = self.scores_data_ptr[j]


    cdef int smooth_best_scores(self, int y) nogil:
        """Stores the smoothed loss gradients and
        smoothed assignments in self.use_off_scores and
        self.use_true_scores respectively if the loss gradient
        is non-negligibly the zero vector.  
        
        Returns 1 if the loss gradient is non-neglibly the zero
        vector otherwise returns 0

        The loss gradient vector and the assignment vector are
        generally sparse and their entries are stored in
        self.use_off_scores, self.use_true_scores with indices
        self.use_off_score_ids, self.use_true_score_ids, respectively
        """
        # need to find the best scores in order to get
        # the normalizations working
        self.find_best_scores(y)
        self.off_score_normalization = (1.0 + self.best_off_score - self.best_true_score)/self.gamma

        if self.off_score_normalization <= LOGEPSILON:
            # negligible loss-gradient
            return 0

        self.off_score_normalization = exp(-self.off_score_normalization)
        self.true_score_normalization = 0.0
        self.n_use_off_scores = 0
        self.n_use_true_scores = 0



        for j in range(self.n_components):
            if self.W_classes_ptr[j] == y:
                self.use_true_scores_ptr[self.n_use_true_scores] = (self.scores_data_ptr[j] - self.best_true_score)/self.beta
                if self.use_true_scores_ptr[self.n_use_true_scores] > LOGEPSILON:
                    self.use_true_scores_ptr[self.n_use_true_scores] = exp(self.use_true_scores_ptr[self.n_use_true_scores])
                    self.true_score_normalization += self.use_true_scores_ptr[self.n_use_true_scores]
                    self.use_true_score_ids_ptr[self.n_use_true_scores] = j
                    self.n_use_true_scores += 1


            else:
                self.use_off_scores_ptr[self.n_use_off_scores] = (self.scores_data_ptr[j] - self.best_off_score)/self.gamma
                
                if self.use_off_scores_ptr[self.n_use_off_scores] > LOGEPSILON:
                    self.use_off_scores_ptr[self.n_use_off_scores] = exp(self.use_off_scores_ptr[self.n_use_off_scores])
                    self.use_off_score_ids_ptr[self.n_use_off_scores] = j
                    self.off_score_normalization += self.use_off_scores_ptr[self.n_use_off_scores]
                    
                    self.n_use_off_scores += 1
        
        return 1

        
    cdef void scale(self, double c) nogil:
        """Scales the weight vector by a constant ``c``. It updates ``wscale``, ``sq_norm``, and ``row_sq_norms``. If ``Wscale`` gets too small we call ``reset_wscale``.
        """
        self.Wscale *= c
        self.sq_norm *= (c*c)
        if self.Wscale < 1e-9:
            self.reset_Wscale()
            
    cdef void reset_Wscale(self) nogil:
        """Scales each coef of ``W`` by ``Wscale`` and resets ``Wscale to 1.
        """
        dscal(<int>self.W.shape[0],
              self.Wscale,
              <double *>self.W.data, 1)
        self.Wscale = 1.0

    cdef double norm(self) nogil:
        """The L2 norm of the weight vector.
        """
        return sqrt(self.sq_norm)
            


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

cdef void _sparse_dot_row_subset(int *X_indices_ptr, int *rownnz,
                                 int* rowstartidx, double *w_ptr, double *z_ptr,
                                 int *use_row_indices,
                                 int n_use_rows) nogil:
    """
    """
    cdef int i,j, idx,row_idx
    cdef int *cur_X_ptr = <int *>X_indices_ptr
    for i in range(n_use_rows):
        row_idx = use_row_indices[i]
        z_ptr[i] = 0.0
        cur_X_ptr = <int *>(& X_indices_ptr[rowstartidx[row_idx]])
        for j in range(rownnz[row_idx]):
            idx = cur_X_ptr[j]
            z_ptr[i] += w_ptr[idx]


def sparse_dot_row_subset(np.ndarray[int,ndim=1,mode='c'] X_indices,
               np.ndarray[int,ndim=1, mode='c'] rownnz,
               np.ndarray[int, ndim=1, mode='c'] rowstartidx,
               np.ndarray[double,ndim=1,mode='c'] w,
                          np.ndarray[int,ndim=1,mode="c"] use_row_indices,
                          int n_use_rows):
    cdef np.ndarray[double,ndim=1,mode='c'] z = np.zeros(w.shape[0],
                                                         dtype=np.float)
    _sparse_dot_row_subset(<int *> X_indices.data, <int *> rownnz.data,
                <int *> rowstartidx.data, 
                <double *>w.data,
                <double *>z.data,
                <int *> use_row_indices.data,
                n_use_rows)
    return z
        
        
    
def sparse_dot(np.ndarray[int,ndim=1,mode='c'] X_indices,
               np.ndarray[int,ndim=1, mode='c'] rownnz,
               np.ndarray[int, ndim=1, mode='c'] rowstartidx,
               np.ndarray[double,ndim=1,mode='c'] w,
               np.ndarray[double,ndim=1,mode='c'] z,
               int X_n_rows):
    _sparse_dot(<int *> X_indices.data, <int *> rownnz.data,
                <int *> rowstartidx.data, 
                X_n_rows,<double *>w.data,
                <double *>z.data)
    
# def multiclass_SAG_smoothed_hinge(
#         np.ndarray[double, ndim=1, mode='c'] weights,
#         np.ndarray[int, ndim=1, mode='c'] weights_classes,
#         np.ndarray[int, ndim=1, mode='c'] weights_components,
#         int n_classes,
#         BinaryArrayDataset, int seed, int n_iter,
#         int shuffle,
#         int verbose,
#         double lambda_param):
#     """
#     """
#     cdef Py_ssize_t n_samples = dataset.n_samples

#     print ("n_samples = %d" % n_samples)

#     cdef MultiWeightMatrix W = MultiWeightMatrix(weights,weights_classes, weights_components, n_classes)

#     print ("sq_norm = %g" % W.sq_norm)
    
#     cdef Py_ssize_t n_components = W.n_components
#     cdef int *x_ind_ptr = NULL
#     cdef short int y = 0
#     cdef int xnnz

#     cdef double eta
#     cdef int i, j


#     # get the SAG initializations
#     cdef np.ndarray[ndim=1,dtype=np.uint8_t,order='c'] visted_node = np.zeros(n_samples,dtype=np.uint8)
#     cdef int num_visted = 0

#     cdef np.ndarray[ndim=1,dtype=double,order='c'] cumulative_steps = np.zeros(n_samples+1,dtype=np.float)
#     cdef np.ndarray[ndim=1,dtype=int,order='c'] prev_non_zero_step = np.zeros(W.n_features,dtype=np.intc)

#     cdef double * cur_data_ptr 

#     for epoch in range(n_iter):
#         if verbose > 0:
#             print ("-- Epoch %d" % epoch)
#         if shuffle > 0:
#             dataset.shuffle(seed)

#         cumulative_steps[0] = cumulative_steps[n_samples]
#         for j in xrange(W.n_features):
#             prev_non_zero_step[j] = 0

#         for i in xrange(W.n_components):
#             cur_data_ptr = <double *>(W.W_data_ptr + i*W.n_features)
#             for j in xrange(W.n_features):
                
        

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel (partial_fit support)
#         Rob Zinkov (passive-aggressive)
#         Lars Buitinck
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


np.import_array()

# Learning rate constants
DEF CONSTANT = 1
DEF OPTIMAL = 2
DEF INVSCALING = 3
DEF PA1 = 4
DEF PA2 = 5

# setup bool types
BOOL = np.uint8
ctypedef np.uint8_t BOOL_t
SHORT = np.int16
ctypedef np.int16_t SHORT_t





cdef class BinaryArrayDataset(object):
    """A binary dataset backed by a two-dimensional binary numpy array

    The dtype of the numpy array is expected to be ``np.uint8``
    and C-style memory layout. Based on the sequential dataset
    in sklearn
    """
    
    cdef int n_samples, current_index
    cdef short int *Y_data_ptr
    cdef int * X_indices_ptr
    cdef int * rowstartidx_ptr
    cdef int * rownnz_ptr
    cdef int * index_data_ptr
    cdef np.ndarray index
    
    def __cinit__(self, np.ndarray[int, ndim=1, mode='c'] X_indices,
                   np.ndarray[int, ndim=1, mode='c'] rownnz,
                   np.ndarray[int, ndim=1, mode='c'] rowstartidx,
                  np.ndarray[SHORT_t,ndim=1, mode='c'] Y):
        """A ``SequentialDataset`` backed by a two-dimensional numpy array.

        Parameters
        ----------
        X : ndarray, dtype=np.uint8, ndim=2, mode='c'
            The samples; a two-dimensional c-contiguous numpy array of dtype np.uint8

        Y : ndarray, dtype=np.int16, ndim=1, mode='c'
            The target classes; a one-dimensional c-contiguous numpy array of dtype double.

        sample_weights : ndarray, dtype=double, ndim=1, mode='c'
            Weight for each sample; a one-dimensional c-continuous numpy array of dtype double
        """
        
            

        self.n_samples = rownnz.shape[0]

        self.current_index = -1
        self.Y_data_ptr = <short int *>Y.data
        self.X_indices_ptr = <int *>X_indices.data
        self.rowstartidx_ptr = <int *>rowstartidx.data
        self.rownnz_ptr = <int *>rownnz.data
        cdef np.ndarray[int, ndim=1,
                        mode='c'] index = np.arange(0,
                                                    self.n_samples,
                                                    dtype=np.intc)
        self.index = index
        self.index_data_ptr = <int *> index.data
        

    cdef void next(self, 
                   int **x_ind_ptr,
                   int *nnz, short int *y) nogil:
            
        cdef int current_index = self.current_index
        if current_index >= (self.n_samples-1):
                current_index = -1

        current_index += 1
        cdef int sample_idx =self.index_data_ptr[current_index]
        cdef int offset = self.rowstartidx_ptr[sample_idx]

        y[0] = self.Y_data_ptr[sample_idx]

        x_ind_ptr[0] = self.X_indices_ptr + offset
        nnz[0] = self.rownnz_ptr[sample_idx]
            
        self.current_index = current_index

    cdef void shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.index)


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

    def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] W,
                  np.ndarray[int, ndim=1, mode='c'] W_classes,
                  np.ndarray[int, ndim=1, mode='c'] W_components,
                  int n_classes):
        """
        W should be put out as a single column vector
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

        self.class_n_components_ptr[W_classes[i-1]] = nrow_components

        self.sq_norm = ddot(self.n_entries,
                                             <double *>W.data, 1, <double *>W.data,1)
            
    
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
            

def multiclass_sgd(np.ndarray[double, ndim=1, mode='c'] weights,
                   np.ndarray[int, ndim=1, mode='c'] weights_classes,
                   np.ndarray[int, ndim=1, mode='c'] weights_components,
                   int n_classes,
                   BinaryArrayDataset dataset, int seed, int n_iter,
                   int shuffle,
                   int verbose,
                   double start_t,
                   double lambda_param,
                   int do_projection):
    cdef Py_ssize_t n_samples = dataset.n_samples

    print n_samples
    cdef MultiWeightMatrix W = MultiWeightMatrix(weights,weights_classes, weights_components, n_classes)

    print ("sq_norm = %g" % W.sq_norm)
    
    cdef Py_ssize_t n_components = W.n_components
    cdef int *x_ind_ptr = NULL
    cdef short int y = 0
    cdef int xnnz, best_true_row, best_off_row
    cdef double best_true_score, best_off_score, scaling
    cdef double eta, t, oneoversqrtlambda
    
    oneoversqrtlambda = 1./sqrt(lambda_param)

    t = start_t + 1.0

    cdef unsigned int epoch
    cdef Py_ssize_t i, j

    for i in range(W.n_classes):
        print ("Class %d in the weight matrix starts on row %d and has %d components" % (i,W.class_row_idx_ptr[i],W.class_n_components_ptr[i]))

    for epoch in range(n_iter):
        if verbose > 0:
            print ("-- Epoch %d" % epoch)
        if shuffle > 0:
            dataset.shuffle(seed)
        for i in range(n_samples):
            if i % 5000 == 0:
                print i
            dataset.next( & x_ind_ptr, & xnnz, & y)

            W.dot( x_ind_ptr, xnnz)
            
            if y == 0:
                best_true_row = 0
                best_off_row = W.class_row_idx_ptr[1]
                
            else:
                best_off_row = 0
                best_true_row = W.class_row_idx_ptr[y]


                
            # initialize the estimated best scores for updates
            best_true_score = W.scores_data_ptr[best_true_row]
            best_off_score = W.scores_data_ptr[best_off_row]
            
            # find the best row for both
            for j in range(W.n_components):

                
                if W.W_classes_ptr[j] == y:
                    # within class 
                    if W.scores_data_ptr[j] > best_true_score:
                        best_true_row = j
                        best_true_score = W.scores_data_ptr[j]
                else:
                    # not within class
                    if W.scores_data_ptr[j] > best_off_score:
                        best_off_row = j
                        best_off_score = W.scores_data_ptr[j]


            scaling = 1./t
            W.scale( (1.0- scaling))
            eta = scaling/lambda_param
            W.binary_add(best_true_row,x_ind_ptr,xnnz,eta)
            eta *= -1.0
            W.binary_add(best_off_row,x_ind_ptr,xnnz,eta)

            if do_projection > 0:
                scaling = oneoversqrtlambda/sqrt(W.sq_norm)
                if scaling < 1.0:
                    W.scale(scaling)

    return W.W
            

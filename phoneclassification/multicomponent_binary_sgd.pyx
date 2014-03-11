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


cdef class FixedPointArrayDataset(object):
    """A binary dataset backed by a two-dimensional binary numpy array

    The dtype of the numpy array is expected to be ``np.uint8``
    and C-style memory layout. Based on the sequential dataset
    in sklearn
    """
    
    cdef int n_samples, current_index
    cdef short int *Y_data_ptr
    cdef int * X_indices_ptr
    cdef np.uint8_t * X_vals_ptr
    cdef double X_val_base
    cdef int * rowstartidx_ptr
    cdef int * rownnz_ptr
    cdef int * index_data_ptr
    cdef np.ndarray index
    
    
    def __cinit__(self, np.ndarray[int, ndim=1, mode='c'] X_indices,
                  np.ndarray[np.uint8_t,ndim=1,mode='c'] X_vals,
                  double X_val_base,
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
        self.X_val_base = X_val_base
        self.Y_data_ptr = <short int *>Y.data
        self.X_indices_ptr = <int *>X_indices.data
        self.X_vals_ptr = <np.uint8_t *>X_vals.data
        self.rowstartidx_ptr = <int *>rowstartidx.data
        self.rownnz_ptr = <int *>rownnz.data
        cdef np.ndarray[int, ndim=1,
                        mode='c'] index = np.arange(0,
                                                    self.n_samples,
                                                    dtype=np.intc)
        self.index = index
        self.index_data_ptr = <int *> index.data
        

    cdef void next(self, 
                   int **x_ind_ptr, np.uint8_t **x_val_ptr,
                   int *nnz, short int *y) nogil:
            
        cdef int current_index = self.current_index
        if current_index >= (self.n_samples-1):
                current_index = -1

        current_index += 1
        cdef int sample_idx =self.index_data_ptr[current_index]
        cdef int offset = self.rowstartidx_ptr[sample_idx]

        y[0] = self.Y_data_ptr[sample_idx]

        x_ind_ptr[0] = self.X_indices_ptr + offset
        x_val_ptr[0] = self.X_vals_ptr + offset
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

    
    cdef void fixed_point_add(self, int w_row,
                         int *x_ind_ptr, np.uint8_t *x_val_ptr, double x_val_base, int xnnz, double c) nogil:
        """Scales fixed point vector sample x by constant c and adds it 
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
        cdef double c_div_Wscale = (c / Wscale)
        cdef double* W_data_ptr 



        
        W_data_ptr =  (self.W_data_ptr + w_row * self.n_features)
        for j in range(xnnz):
            idx = x_ind_ptr[j]
            val = (<double> x_val_ptr[j])/x_val_base
            innerprod += (W_data_ptr[idx] * val)
            xsqnorm += val * val
            W_data_ptr[idx] += val * c_div_Wscale


        self.sq_norm += (xsqnorm * c * c) + (2.0 * innerprod * Wscale * c)

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

            scores_data_ptr[i] *= self.Wscale

    cdef void fixed_point_dot(self, int *x_ind_ptr, np.uint8_t *x_val_ptr, double x_val_base,
                    int xnnz) nogil:
        """Computes the matrix-vector dot product between a sample x and the weight matrix. And stores the result in the weight vector.
        fixed_point 

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
                scores_data_ptr[i] += cur_data_ptr[idx] * (<double> x_val_ptr[j])
                
            scores_data_ptr[i] *= self.Wscale/ x_val_base

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
            

def multiclass_sgd(np.ndarray[double, ndim=1, mode='c'] weights,
                   np.ndarray[int, ndim=1, mode='c'] weights_classes,
                   np.ndarray[int, ndim=1, mode='c'] weights_components,
                   int n_classes,
                   BinaryArrayDataset dataset, int seed, int n_iter,
                   int shuffle,
                   int verbose,
                   double start_t,
                   double lambda_param,
                   int do_projection, double time_scaling, int use_hinge):
    cdef Py_ssize_t n_samples = dataset.n_samples

    print ("n_samples = %d " %n_samples)
    cdef MultiWeightMatrix W = MultiWeightMatrix(weights,weights_classes, weights_components, n_classes, 1.0, 1.0)

    print ("sq_norm = %g" % W.sq_norm)
    
    cdef Py_ssize_t n_components = W.n_components
    cdef int *x_ind_ptr = NULL
    cdef np.uint8_t *x_val_ptr = NULL
    cdef short int y = 0
    cdef int xnnz, best_true_row, best_off_row, do_update
    cdef double best_true_score, best_off_score, scaling
    cdef double eta, t, oneoversqrtlambda
    
    oneoversqrtlambda = 1./sqrt(lambda_param)

    t = start_t + 1.0
    print ("t=%g" % t)

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
            if verbose > 1:
                print ("\n\n-----Working on sample %d------" % i)
            if i % 5000 == 0:
                print i
            dataset.next( & x_ind_ptr, & xnnz, & y)

            W.dot( x_ind_ptr, xnnz)
            if verbose > 1:
                print ("y = %d" % y)
                for j in range(xnnz):
                    print ("x[%d] = 1" % x_ind_ptr[j])

            if y == 0:
                best_true_row = 0
                best_off_row = W.class_row_idx_ptr[1]
                
            else:
                best_off_row = 0
                best_true_row = W.class_row_idx_ptr[y]

            if verbose > 1:
                print ("Initial best_true_row= %d\t best_off_row=%d" % (best_true_row,best_off_row))
                
            # initialize the estimated best scores for updates
            best_true_score = W.scores_data_ptr[best_true_row]
            best_off_score = W.scores_data_ptr[best_off_row]
            
            # find the best row for both
            for j in range(W.n_components):

                
                if W.W_classes_ptr[j] == y:
                    if verbose > 1:
                        print ("Within class Comparing within class on row %d with score %g " % (j,W.scores_data_ptr[j]))
                    # within class 
                    if W.scores_data_ptr[j] > best_true_score:
                        
                        if verbose > 1:
                            print ("row %d score %g is bigger than previous max %g " % (j,W.scores_data_ptr[j], best_true_score))
                        best_true_row = j
                        best_true_score = W.scores_data_ptr[j]
                        if verbose > 1:
                            print ("best_true_row=%d  best_true_score=%g" % (best_true_row,best_true_score))
                else:
                    if verbose > 1:
                        print ("Comparing outside class on row %d with score %g " % (j,W.scores_data_ptr[j]))
                    # not within class
                    if W.scores_data_ptr[j] > best_off_score:
                        if verbose > 1:
                            print ("Outside class: row %d score %g is bigger than previous max %g " % (j,W.scores_data_ptr[j], best_off_score))
                            
                        best_off_row = j
                        best_off_score = W.scores_data_ptr[j]
                        if verbose > 1:
                            print ("best_off_row=%d  best_off_score=%g" % (best_off_row,best_off_score))

                            
                            
            if use_hinge > 0:
                if best_off_score - best_true_score > -1:
                    do_update = 1
                else:
                    do_update = 0
            else:
                if (best_off_score - best_true_score > -1) and (
                        best_off_score -best_true_score < 1):
                    do_update = 1
                else:
                    do_update = 0


            scaling = 1./(t+(<double>i)/time_scaling)
            W.scale( (1.0- scaling))
            if verbose > 1:
                print ("1-scaling=%g\t W.scaling=%g" % (1.0-scaling,W.Wscale))

            if do_update > 0:
            
                if verbose > 1:
                    print ("Loss exceeded -1-- doing update -W.scaling=%g" % W.Wscale)
                eta = scaling/lambda_param
                if verbose > 1:
                    print ("eta=%g" % eta)
                if verbose > 1:
                    for j in range(W.n_features):
                        print ("W[%d,%d] = %g" % (best_true_row,j,
                                              (W.W_data_ptr + best_true_row * W.n_features)[j]*W.Wscale))


                W.binary_add(best_true_row,x_ind_ptr,xnnz,eta)
                if verbose > 1:
                    for j in range(W.n_features):
                        print ("W[%d,%d] = %g" % (best_true_row,j,
                                              (W.W_data_ptr + best_true_row * W.n_features)[j]*W.Wscale))

                    print " "
                               
                eta *= -1.0
                if verbose > 1:
                    for j in range(W.n_features):
                        print ("W[%d,%d] = %g" % (best_off_row,j,
                                              (W.W_data_ptr + best_off_row * W.n_features)[j]*W.Wscale))



                W.binary_add(best_off_row,x_ind_ptr,xnnz,eta)
                if verbose > 1:
                    for j in range(W.n_features):
                        print ("W[%d,%d] = %g" % (best_off_row,j,
                                              (W.W_data_ptr + best_off_row * W.n_features)[j]*W.Wscale))


            if do_projection > 0:
                scaling = oneoversqrtlambda/sqrt(W.sq_norm)
                if scaling < 1.0:
                    W.scale(scaling)


    cdef np.ndarray[ndim=2,dtype=double] out_weights = np.zeros((W.n_components,W.n_features),dtype=np.float)
    cdef double * cur_data_ptr = W.W_data_ptr
    for i in range(W.n_components):
        cur_data_ptr = <double *>(W.W_data_ptr + i*W.n_features)
        for j in range(W.n_features):
            out_weights[i,j] = cur_data_ptr[j] * W.Wscale


    return out_weights
            

def multiclass_sgd_fixed_point(np.ndarray[double, ndim=1, mode='c'] weights,
                   np.ndarray[int, ndim=1, mode='c'] weights_classes,
                   np.ndarray[int, ndim=1, mode='c'] weights_components,
                   int n_classes,
                               FixedPointArrayDataset dataset, int seed, int n_iter,
                   int shuffle,
                   int verbose,
                   double start_t,
                   double lambda_param,
                   int do_projection, double time_scaling, int use_hinge):
    cdef Py_ssize_t n_samples = dataset.n_samples

    print ("n_samples = %d " %n_samples)
    cdef MultiWeightMatrix W = MultiWeightMatrix(weights,weights_classes, weights_components, n_classes, 1.0, 1.0)

    print ("sq_norm = %g" % W.sq_norm)
    
    cdef Py_ssize_t n_components = W.n_components
    cdef int *x_ind_ptr = NULL
    cdef np.uint8_t *x_val_ptr = NULL
    cdef short int y = 0
    cdef double x_val_base = dataset.X_val_base
    cdef int xnnz, best_true_row, best_off_row, do_update
    cdef double best_true_score, best_off_score, scaling
    cdef double eta, t, oneoversqrtlambda
    
    oneoversqrtlambda = 1./sqrt(lambda_param)

    t = start_t + 1.0
    print ("t=%g" % t)

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
            if verbose > 1:
                print ("\n\n-----Working on sample %d------" % i)
            if i % 5000 == 0:
                print i
            dataset.next( & x_ind_ptr, & x_val_ptr, & xnnz, & y)

            W.fixed_point_dot( x_ind_ptr, x_val_ptr, x_val_base, xnnz)
            if verbose > 1:
                print ("y = %d" % y)
                for j in range(xnnz):
                    print ("x[%d] = 1" % x_ind_ptr[j])

            if y == 0:
                best_true_row = 0
                best_off_row = W.class_row_idx_ptr[1]
                
            else:
                best_off_row = 0
                best_true_row = W.class_row_idx_ptr[y]

            if verbose > 1:
                print ("Initial best_true_row= %d\t best_off_row=%d" % (best_true_row,best_off_row))
                
            # initialize the estimated best scores for updates
            best_true_score = W.scores_data_ptr[best_true_row]
            best_off_score = W.scores_data_ptr[best_off_row]
            
            # find the best row for both
            for j in range(W.n_components):

                
                if W.W_classes_ptr[j] == y:
                    if verbose > 1:
                        print ("Within class Comparing within class on row %d with score %g " % (j,W.scores_data_ptr[j]))
                    # within class 
                    if W.scores_data_ptr[j] > best_true_score:
                        
                        if verbose > 1:
                            print ("row %d score %g is bigger than previous max %g " % (j,W.scores_data_ptr[j], best_true_score))
                        best_true_row = j
                        best_true_score = W.scores_data_ptr[j]
                        if verbose > 1:
                            print ("best_true_row=%d  best_true_score=%g" % (best_true_row,best_true_score))
                else:
                    if verbose > 1:
                        print ("Comparing outside class on row %d with score %g " % (j,W.scores_data_ptr[j]))
                    # not within class
                    if W.scores_data_ptr[j] > best_off_score:
                        if verbose > 1:
                            print ("Outside class: row %d score %g is bigger than previous max %g " % (j,W.scores_data_ptr[j], best_off_score))
                            
                        best_off_row = j
                        best_off_score = W.scores_data_ptr[j]
                        if verbose > 1:
                            print ("best_off_row=%d  best_off_score=%g" % (best_off_row,best_off_score))

                            
                            
            if use_hinge > 0:
                if best_off_score - best_true_score > -1:
                    do_update = 1
                else:
                    do_update = 0
            else:
                if (best_off_score - best_true_score > -1) and (
                        best_off_score -best_true_score < 1):
                    do_update = 1
                else:
                    do_update = 0


            scaling = 1./(t+(<double>i)/time_scaling)
            W.scale( (1.0- scaling))
            if verbose > 1:
                print ("1-scaling=%g\t W.scaling=%g" % (1.0-scaling,W.Wscale))

            if do_update > 0:
            
                if verbose > 1:
                    print ("Loss exceeded -1-- doing update -W.scaling=%g" % W.Wscale)
                eta = scaling/lambda_param
                if verbose > 1:
                    print ("eta=%g" % eta)
                if verbose > 1:
                    for j in range(W.n_features):
                        print ("W[%d,%d] = %g" % (best_true_row,j,
                                              (W.W_data_ptr + best_true_row * W.n_features)[j]*W.Wscale))


                W.fixed_point_add(best_true_row,x_ind_ptr, x_val_ptr, x_val_base, xnnz,eta)
                if verbose > 1:
                    for j in range(W.n_features):
                        print ("W[%d,%d] = %g" % (best_true_row,j,
                                              (W.W_data_ptr + best_true_row * W.n_features)[j]*W.Wscale))

                    print " "
                               
                eta *= -1.0
                if verbose > 1:
                    for j in range(W.n_features):
                        print ("W[%d,%d] = %g" % (best_off_row,j,
                                              (W.W_data_ptr + best_off_row * W.n_features)[j]*W.Wscale))



                W.fixed_point_add(best_true_row,x_ind_ptr, x_val_ptr, x_val_base, xnnz,eta)
                if verbose > 1:
                    for j in range(W.n_features):
                        print ("W[%d,%d] = %g" % (best_off_row,j,
                                              (W.W_data_ptr + best_off_row * W.n_features)[j]*W.Wscale))


            if do_projection > 0:
                scaling = oneoversqrtlambda/sqrt(W.sq_norm)
                if scaling < 1.0:
                    W.scale(scaling)


    cdef np.ndarray[ndim=2,dtype=double] out_weights = np.zeros((W.n_components,W.n_features),dtype=np.float)
    cdef double * cur_data_ptr = W.W_data_ptr
    for i in range(W.n_components):
        cur_data_ptr = <double *>(W.W_data_ptr + i*W.n_features)
        for j in range(W.n_features):
            out_weights[i,j] = cur_data_ptr[j] * W.Wscale


    return out_weights
            


def multiclass_smoothed_sgd(np.ndarray[double, ndim=1, mode='c'] weights,
                   np.ndarray[int, ndim=1, mode='c'] weights_classes,
                   np.ndarray[int, ndim=1, mode='c'] weights_components,
                   int n_classes,
                   BinaryArrayDataset dataset, int seed, int n_iter,
                   int shuffle,
                   int verbose,
                   double start_t,
                   double lambda_param,
                            int do_projection, double time_scaling,
                            double gamma, double beta):
    cdef Py_ssize_t n_samples = dataset.n_samples

    print n_samples
    cdef MultiWeightMatrix W = MultiWeightMatrix(weights,weights_classes, weights_components, n_classes, gamma, beta)

    print ("sq_norm = %g" % W.sq_norm)
    
    cdef Py_ssize_t n_components = W.n_components
    cdef int *x_ind_ptr = NULL
    cdef short int y = 0
    cdef int xnnz, best_true_row, best_off_row
    cdef double best_true_score, best_off_score, scaling, add_scaling
    cdef double off_score_normalization, true_score_normalization
    #
    cdef int n_use_off_scores, n_use_true_scores, use_off_scores_idx, use_true_scores_idx
    cdef np.ndarray[ndim=1,dtype=int] use_off_score_ids = np.zeros(n_components,dtype=np.intc)
    cdef np.ndarray[ndim=1,dtype=double] use_off_scores = np.zeros(n_components,dtype=np.float)
    cdef np.ndarray[ndim=1,dtype=int] use_true_score_ids = np.zeros(n_components,dtype=np.intc)
    cdef np.ndarray[ndim=1,dtype=double] use_true_scores = np.zeros(n_components,dtype=np.float)
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
            if verbose > 1:
                print ("\n\n-----Workign on sample %d------" % i)
            if i % 5000 == 0:
                print i
            dataset.next( & x_ind_ptr, & xnnz, & y)

            W.dot( x_ind_ptr, xnnz)
            if verbose > 1:
                print ("y = %d" % y)
            if y == 0:
                best_true_row = 0
                best_off_row = W.class_row_idx_ptr[1]
                
            else:
                best_off_row = 0
                best_true_row = W.class_row_idx_ptr[y]

            if verbose > 1:
                print ("Initial best_true_row= %d\t best_off_row=%d" % (best_true_row,best_off_row))
                
            # initialize the estimated best scores for updates
            best_true_score = W.scores_data_ptr[best_true_row]
            best_off_score = W.scores_data_ptr[best_off_row]
            
            # find the best row for both
            for j in range(W.n_components):

                
                if W.W_classes_ptr[j] == y:
                    if verbose > 1:
                        print ("Within class Comparing within class on row %d with score %g " % (j,W.scores_data_ptr[j]))
                    # within class 
                    if W.scores_data_ptr[j] > best_true_score:
                        
                        if verbose > 1:
                            print ("row %d score %g is bigger than previous max %g " % (j,W.scores_data_ptr[j], best_true_score))
                        best_true_row = j
                        best_true_score = W.scores_data_ptr[j]
                        if verbose > 1:
                            print ("best_true_row=%d  best_true_score=%g" % (best_true_row,best_true_score))
                else:
                    if verbose > 1:
                        print ("Comparing outside class on row %d with score %g " % (j,W.scores_data_ptr[j]))
                    # not within class
                    if W.scores_data_ptr[j] > best_off_score:
                        if verbose > 1:
                            print ("Outside class: row %d score %g is bigger than previous max %g " % (j,W.scores_data_ptr[j], best_off_score))
                            
                        best_off_row = j
                        best_off_score = W.scores_data_ptr[j]
                        if verbose > 1:
                            print ("best_off_row=%d  best_off_score=%g" % (best_off_row,best_off_score))

            # first part of update is just a rescaling of the weights

            scaling = 1./(t+<double>i)
            W.scale( (1.0- scaling))
            if verbose > 1:
                print ("scaling=%g\t W.scaling=%g" % (scaling,W.Wscale))


            # now we find whether to do an update
            if (1.0 + best_off_score - best_true_score)/gamma > LOGEPSILON:

                off_score_normalization = exp(-(1.0+best_off_score-best_true_score)/gamma)
                true_score_normalization = 0.0
                n_use_off_scores = 0
                n_use_true_scores = 0
                # find the best row for both
                for j in range(W.n_components):            
                    if W.W_classes_ptr[j] == y:
                        # if verbose > 1:
                        #     print ("Within class Comparing within class on row %d with score %g " % (j,W.scores_data_ptr[j]))
                        # within class 
                        add_scaling = (W.scores_data_ptr[j] - best_true_score)/beta
                        if add_scaling  > LOGEPSILON:
                            use_true_scores[n_use_true_scores] = exp(add_scaling)
                            true_score_normalization += use_true_scores[n_use_true_scores]
                            use_true_score_ids[n_use_true_scores] = j
                            n_use_true_scores += 1
                            
                            # if verbose > 1:
                            #     print ("row %d score %g is bigger than previous max %g " % (j,W.scores_data_ptr[j], best_true_score))
                    else:
                        # if verbose > 1:
                        #     print ("Comparing outside class on row %d with score %g " % (j,W.scores_data_ptr[j]))
                        # not within class
                        add_scaling = (W.scores_data_ptr[j] - best_off_score )/gamma
                        if add_scaling > LOGEPSILON:
                            # if verbose > 1:
                            #     print ("Outside class: row %d score %g is bigger than previous max %g " % (j,W.scores_data_ptr[j], best_off_score))
                            use_off_scores[n_use_off_scores] = exp(add_scaling)
                            use_off_score_ids[n_use_off_scores] = j
                            off_score_normalization += use_off_scores[n_use_off_scores]
                            n_use_off_scores += 1
                                
                            # if verbose > 1:
                            #     print ("best_off_row=%d  best_off_score=%g" % (best_off_row,best_off_score))
                    
                eta = scaling/lambda_param
                for j in range(n_use_true_scores):
                    add_scaling = eta*use_true_scores[j]
                    if add_scaling > EPSILON * true_score_normalization:
                        W.binary_add(use_true_score_ids[j],
                                     x_ind_ptr,xnnz,add_scaling/true_score_normalization)


                for j in range(n_use_off_scores):
                    add_scaling = eta*use_off_scores[j]
                    if add_scaling > EPSILON * off_score_normalization:
                        W.binary_add(use_off_score_ids[j],
                                 x_ind_ptr,xnnz,-add_scaling/off_score_normalization)

                            

            if do_projection > 0:
                scaling = oneoversqrtlambda/sqrt(W.sq_norm)
                if scaling < 1.0:
                    W.scale(scaling)


    cdef np.ndarray[ndim=2,dtype=double] out_weights = np.zeros((W.n_components,W.n_features),dtype=np.float)
    cdef double * cur_data_ptr = W.W_data_ptr
    for i in range(W.n_components):
        cur_data_ptr = <double *>(W.W_data_ptr + i*W.n_features)
        for j in range(W.n_features):
            out_weights[i,j] = cur_data_ptr[j] * W.Wscale
    return out_weights

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
                
        

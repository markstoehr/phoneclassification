import numpy as np

def binary_to_bsparse(X):
    """Convert a binary 2d array into a list feature indices.
    Output is three 1d arrays

    Returns
    -------
    feature_ind : np.ndarray shape (nnz,2)
        Each entry is an index to a row and a column with a non-zero
        entry

    rownnz : np.ndarray shape(nrows)
        Each entry is the number of non-zero entries in a given
        row of the original matrix
  
    rowstartidx : np.ndarray shape(nrows+1)
        Each entry is the start index of the next row the final
        entry is just nnz and the first entry is 0
    """
    feature_ind = np.array(np.where(X),dtype=np.intc).T
    rownnz = (X > 0).sum(1).astype(np.intc)
    rowstartidx = np.zeros(X.shape[0]+1,dtype=np.intc)
    rowstartidx[1:] = np.cumsum(rownnz)

    return feature_ind, rownnz, rowstartidx

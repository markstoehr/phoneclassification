from __future__ import division
import numpy as np
from scipy.ndimage.filters import maximum_filter

def extract_patches(X,patch_shape=(5,5,8)):
    """
    Extract all patches from X using patch_shape
    that's patch_shape
    Implemented using stride_tricks
    """
    try:
        patch_view_shape=tuple(np.subtract(X.shape,patch_shape)+1)[:-1]+patch_shape
    except:
        import pdb; pdb.set_trace()
    patch_arr_view = np.lib.stride_tricks.as_strided(X,
                                                     patch_view_shape,
                                                     X.strides[:-1]+X.strides)
    return patch_arr_view

def extract_spec_patches(X,patch_shape=(5,5)):
    """
    Extract all patches from X using patch_shape
    that's patch_shape
    Implemented using stride_tricks
    for spectrograms
    """
    try:
        patch_view_shape=tuple(np.subtract(X.shape,patch_shape)+1)+patch_shape
    except:
        import pdb; pdb.set_trace()
    patch_arr_view = np.lib.stride_tricks.as_strided(X,
                                                     patch_view_shape,
                                                     X.strides+X.strides)
    return patch_arr_view

def get_patch_matrix(patch_arr_view):
    """
    flatten patches into a matrix
    """
    return np.lib.stride_tricks.as_strided(patch_arr_view.ravel(),
                                    (np.prod(patch_arr_view.shape[:2]),
                                     np.prod(patch_arr_view.shape[2:])))

def code_spread_parts(X,flat_part_log_odds,constants,part_shape,spread_neighborhood,
               count_threshold=10,likelihood_threshold=-np.inf):
    """
    Parameters:
    ============
    X:
       Data matrix

    flat_part_log_odds:
       Two dimensional matrix containing the flattened log odds for the parts

    contants:
       constants for computing the log-likelihood (log partition function for the part models)

    part_shape:
       tuple giving the original shape for the parts

    spread_neighborhood:
       neighborhood to spread over

    count_threshold:
       minimum counts for edges in a patch for it to be used
    
    
    """
    patch_arr_view = extract_patches(X,part_shape)
    patch_counts = patch_arr_view.sum(-1).sum(-1).sum(-1)
    nparts = len(constants)
    try:
        part_likelihoods = np.lib.stride_tricks.as_strided(
        np.dot(get_patch_matrix(patch_arr_view),
               flat_part_log_odds) + constants,
            patch_arr_view.shape[:2] + (nparts,),
        (patch_arr_view.shape[1] * constants.strides[0]*nparts,
         constants.strides[0]*nparts,
         constants.strides[0]))
    except:
        import pdb; pdb.set_trace()
    max_likes = part_likelihoods.max(-1)
    max_likes[max_likes < likelihood_threshold] = 1
    max_likes[patch_counts < count_threshold] = 1
    part_codes = (part_likelihoods >= np.lib.stride_tricks.as_strided(max_likes,
                                                                     part_likelihoods.shape, max_likes.strides + (0,))).astype(np.uint8)
    
    return maximum_filter(part_codes,size=spread_neighborhood+(1,),cval=0,mode='constant')
                                         
                                          
                                         

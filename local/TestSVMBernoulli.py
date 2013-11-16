from __future__ import division

import numpy as np
import argparse

def get_mapping_coefs_intercepts(labels,component_ids,fl_prefices):
    """
    Get the SVM coefficient matrix containing each SVM as a column,
    the intercepts as a vector and a matrix that maps
    label index and the component index to an index of the svm
    column that should be used.

    """
    label_component_to_svm_id = np.zeros( (np.max(labels)+1,
                                           np.max(component_ids)+1),
                                          dtype=int)
    
    svm0 = np.load('%s_coef.npy' % fl_prefices[0])
    svm_length = len(svm0)
    coefs = np.zeros((len(fl_prefices),
                      svm_length))
    intercepts = np.zeros(len(fl_prefices))
    for fl_id, fl_path in enumerate(fl_prefices):
        coefs[fl_id] = np.load('%s_coef.npy' % fl_path)
        intercepts[fl_id] = np.load('%s_intercept.npy' % fl_path)[0]
        try:
            label_component_to_svm_id[labels[fl_id],
                                  component_ids[fl_id]] = fl_id
        except: import pdb; pdb.set_trace()
        
                                           
    return label_component_to_svm_id, coefs.T,intercepts

def get_bernoulli_templates(template_list):
    """
    Load in the templates and get the log odds filters,
    the constant terms and the mapping from identities to phones
    and template components
    """
    template0 = np.load(template_list[0])
    template_size = template0[0].size
    templates = []
    template_phn_ids = []
    template_component_ids = []
    for t_id, tpath in enumerate(template_list):
        template = np.load(tpath)
        templates.extend(template.reshape(template.shape[0],
                                          template_size))
        template_phn_ids.extend( template.shape[0] * [t_id])
        template_component_ids.extend( range(template.shape[0]))
        
    templates = np.array(templates)
    template_phn_ids = np.array(template_phn_ids)
    template_component_ids = np.array(template_component_ids)

    log_inv_templates = np.log(1-templates)
    log_odds = (np.log(templates) - log_inv_templates).T
    constants = log_inv_templates.sum(-1)
        
    return log_odds, constants, template_phn_ids, template_component_ids

def main(args):
    """
    """
    # load in the svm
    label_component_to_svm_id, coefs, intercepts = get_mapping_coefs_intercepts(np.loadtxt(args.svm_labels_fl),
                                                                                np.loadtxt(args.svm_component_ids_fl),
                                                                                open(args.svm_fl_prefices_fl,'r').read().strip().split('\n'))

    # load in the Bernoulli templates
    log_odds, constants, template_phn_ids, template_component_ids = get_bernoulli_templates(args.templates)

    
    for phn_id, data_path in enumerate(args.data):
        # load in the data
        X = np.load(data_path)
        X_shape = X.shape[1:]
        X = X.reshape(len(X),np.prod(X_shape))
        
        # get the scores with the svm over the labels
        scores = np.dot(X,log_odds) + constants
        top_predicted_classes = template_phn_ids[np.argsort(-scores,1).ravel()].reshape(scores.shape)
        top_predicted_components = template_component_ids[np.argsort(-scores,1).ravel()].reshape(scores.shape)

        scores = np.dot(X,coefs) + intercepts
        Z = label_component_to_svm_id[top_predicted_classes.ravel(),top_predicted_components.ravel()].reshape(scores.shape)
        row_idx=np.arange(Z.shape[0]*5)/5
        top_scores = scores[row_idx,Z[:,:5].ravel()].reshape(Z[:,:5].shape)
        top_score_indices = (1+top_scores.shape[1]*(top_scores > 0).astype(int)) * np.lib.stride_tricks.as_strided(np.arange(top_scores.shape[1])+1,shape=top_scores.shape,strides=(0,8)) -1
        first_svm_detection_index = np.argmin(top_score_indices,1)
        is_detection = np.min(top_score_indices,1) < top_scores.shape[1]
        top_predicted_classes[np.arange(len(top_predicted_classes)),
                              first_svm_detection_index]
        true_classifications = (top_scores < 0).astype(int) * 
        (top_predicted_classes == phn_id)
        
        import pdb; pdb.set_trace()


if __name__=='__main__':
    parser = argparse.ArgumentParser("""Compute label predictions using
    a combination of a likelihood ratio test and an svm
    """)
    parser.add_argument('--data',
                        type=str,
                        nargs='+',
                        help='data files to load in for testing, should be one per phone class and the ordering of the data file should reflect the ordering of the data')
    parser.add_argument('--templates',
                        type=str,
                        nargs='+',
                        help='template files, one per phone class, should be in the same order as the data files')
    parser.add_argument('--svm_labels_fl',
                        type=str,
                        help='path to the file containing where the labels for the svms are saved')
    parser.add_argument('--svm_component_ids_fl',
                        type=str,
                        help='path to where the component identities associated with the svms are saved')
    parser.add_argument('--svm_fl_prefices_fl',
                        type=str,
                        help='path to where the prefices are saved for the files where the _coef.npy and _intercept.py files are saved contained the svm learned parameters')
    main(parser.parse_args())

import numpy as np
import argparse, itertools

def out_ids_to_svm_scorer(out_ids,num_phns):
    """
    """

    svm_scorer = np.zeros((len(out_ids),num_phns))
    num_svms = len(out_ids)
    for i,out_id in enumerate(out_ids):
        svm_scorer[i,out_id[1]] = -1
        svm_scorer[i,out_id[2]] = 1

    return svm_scorer

def get_leehon_matrix(leehon_mapping_dict,
                      phn_ids):
    phn_range = list(np.sort(np.unique(leehon_mapping_dict.values())))
    for p_id, p in enumerate(phn_range):
        print p_id,p
    leehon_matrix = np.zeros((len(phn_ids),
                              len(phn_range)))
    for phn_id, phn in enumerate(phn_ids):
        mapped_phone = leehon_mapping_dict[phn]
        mapped_idx = phn_range.index(mapped_phone)
        print phn_id, phn, mapped_phone, mapped_idx

        leehon_matrix[phn_id,mapped_idx] = 1

    return leehon_matrix

def main(args):
    """
    """
    phn_ids = [ l.split()[1] for l in open(args.phn_ids,'r').read().strip().split('\n')]
    leehon_matrix = get_leehon_matrix(
        dict(tuple( l.split() for l in open(args.leehon_mapping,'r').read().strip().split('\n'))),
        phn_ids)

    coefs = np.load(args.coefs)
    intercepts = np.load(args.intercepts)
    out_ids = [(lambda x: (int(x[0]),
                           int(x[1]),
                           int(x[2]),
                           ))(k.split())
               for k in open(args.out_ids,'r')]

    input_ids = np.array(args.input_ids)
    num_phns = len(phn_ids)

    svm_scorer = out_ids_to_svm_scorer(out_ids,num_phns).T
    num_test_phns = len(args.input_data_matrices)
    confusion_matrix = np.zeros((num_test_phns,
                                 num_phns))


    for test_phn_idx, (phn_test_fl, phn_test_length_fl) in enumerate(itertools.izip(args.input_data_matrices,                                                                            args.input_lengths)):
        X0 = np.load(phn_test_fl).astype(np.uint8)
        X0_shape = X0.shape[1:]
        X = np.zeros( (len(X0), np.prod(X0_shape)+2),dtype=float)
        X[:,:np.prod(X0_shape)] = X0.reshape(len(X0),
                                             np.prod(X0_shape))
        ls = np.log(np.loadtxt(phn_test_length_fl,dtype=int)[:,2])

        X[:,-2] = ls
        X[:,-1] = ls * ls
        Y = (np.dot(X,coefs.T) + intercepts)
        vote_vec = np.zeros((len(Y),num_phns))
        for svm_phn_id, svm_score_vec in enumerate(svm_scorer):
            vote_vec[:,svm_phn_id]=(Y * svm_score_vec > 0).sum(1)

        confusion_matrix[test_phn_idx] = np.bincount(vote_vec.argmax(1),minlength=num_phns)


    leehon_confusion_matrix = np.dot(leehon_matrix.T,
                                     np.dot(confusion_matrix,
                                            leehon_matrix))
    np.save(args.out_confusion_matrix,confusion_matrix)
    np.save(args.out_leehon_confusion_matrix,leehon_confusion_matrix)

    print np.diag(leehon_confusion_matrix).sum()/leehon_confusion_matrix.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""
    Test the svm on data
    """)
    parser.add_argument('--coefs',type=str,
                        help='path to the folder containing the SVM linear filters')
    parser.add_argument('--intercepts',type=str,
                        help='path to the file containing the intercepts')
    parser.add_argument('--out_ids',type=str,
                        help='path to the metadata that shows which entry of the intercept vector and row of coefs cooresponds to what pair of phone comparison SVM')
    parser.add_argument('--input_data_matrices',
                        type=str,
                        nargs='+',
                        help='list of paths to input matrices, generally assumed to only accept two')
    parser.add_argument('--input_lengths',
                        type=str,
                        nargs='+',
                        default=None,
                        help='list of paths to the lengths of the input examples of training data, Default is None in which case nothing is included')
    parser.add_argument('--input_ids',
                        type=int,
                        nargs='+',
                        help='identities for the phones in questions')
    parser.add_argument('--out_confusion_matrix',type=str,
                        help='path of where to save the confusion matrix--which has the raw counts for the confusions')
    parser.add_argument('--out_leehon_confusion_matrix',type=str,
                        help='path of where to save the confusion matrix--which has the leehon adjusted counts for the confusions')
    parser.add_argument('--leehon_mapping',
                        type=str,
                        default=None,
                        help='path to where the leehon mapping is contained')
    parser.add_argument('--phn_ids',
                        type=str,
                        help='path to the file containing the identities for each of the phones')
    parser.add_argument('--top_predicted_labels',
                        type=str,
                        default=None,
                        help='if included makes it so that the voting is only held between a small number of the components')
    main(parser.parse_args())

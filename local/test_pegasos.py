from __future__ import division
import numpy as np
import argparse, itertools
from template_speech_rec import configParserWrapper
from CRunSVM import get_leehon_matrix


def main(args):
    """
    get all the data matrices and process the data
    """
    phones = np.loadtxt(args.phones,dtype=str)

    leehon_mapping_dict = dict(tuple( l.split() for l in open(args.leehon_mapping,'r').read().strip().split('\n')))


    
    leehon_matrix = get_leehon_matrix(
        leehon_mapping_dict,
        phones)


    model_list = np.loadtxt(args.model_list,dtype='str')
    w_dim = np.load(model_list[0][-1]).size
    pairwise_classifiers = np.zeros((len(model_list),
                                     w_dim))

    pairwise_meta = np.zeros((len(model_list),2),dtype=int)
    phone_pair_to_classifier = {}
    for i, (phone1,phone2,w_path) in enumerate(model_list):
        pairwise_classifiers[i] = np.load(w_path).ravel()
        phone_pair_to_classifier[(int(phone1),
                                  int(phone2))] = i
        pairwise_meta[i,0] = int(phone1)
        pairwise_meta[i,1] = int(phone2)


    pairwise_classifiers = pairwise_classifiers.T
    confusion_matrix = np.zeros((len(phones),len(phones)))
    known_pair_confusion_matrix = np.zeros((len(phones),len(phones)))
    for phone_id, phone in enumerate(phones):
        print phone_id
        N_X = np.load('%s/%s_%s' % ( args.data_prefix,
                                   phone,
                                   args.data_suffix)).shape[0]
        X = np.ones((N_X,w_dim))
        X[:,:-1] = np.load('%s/%s_%s' % ( args.data_prefix,
                                   phone,
                                   args.data_suffix)).reshape(
                                       N_X,w_dim-1)

        scores = np.dot(X,pairwise_classifiers)

        max_votes = np.zeros(len(X),dtype=int)
        max_vote_id = np.zeros(len(X),dtype=int)
        for classify_phone_id , classify_phone in enumerate(phones):
            if classify_phone_id < phone_id:
                known_pair_components = (pairwise_meta[:,0] == classify_phone_id) * (pairwise_meta[:,1] == phone_id)
                if known_pair_components.sum() > 1:
                    import pdb; pdb.set_trace()

                max_pair_scores = scores[:,known_pair_components].max(-1)
                
                known_pair_confusion_matrix[
                    phone_id,
                    classify_phone_id] = (max_pair_scores > 0).sum()
                
            elif classify_phone_id > phone_id:
                known_pair_components = (pairwise_meta[:,1] == classify_phone_id) * (pairwise_meta[:,0] == phone_id)
                if known_pair_components.sum() > 1:
                    import pdb; pdb.set_trace()

                max_pair_scores = - scores[:,known_pair_components].min(-1)
                # number of mistakes
                known_pair_confusion_matrix[
                    phone_id,
                    classify_phone_id] = (max_pair_scores > 0).sum()
                
            else:
                known_pair_confusion_matrix[
                    phone_id,
                    phone_id] = len(X)

            

            pos_components = pairwise_meta[:,0] == classify_phone_id
            neg_components = pairwise_meta[:,1] == classify_phone_id

            votes = (scores[:,pos_components] > 0).astype(int).sum(-1)
            votes += (scores[:,neg_components]< 0).sum(-1)
            
            max_vote_id[max_votes <= votes] = classify_phone_id
            max_votes = np.maximum(max_votes, votes)
   

        confusion_matrix[phone_id] = np.bincount(max_vote_id,minlength=len(phones))
        print confusion_matrix[phone_id,phone_id]/confusion_matrix[phone_id].sum()
        

    leehon_confusion_matrix = np.dot(leehon_matrix.T,
                                     np.dot(confusion_matrix,
                                            leehon_matrix))        

    np.save(args.out_confusion_matrix,confusion_matrix)
    np.save(args.out_known_pair_confusion_matrix,known_pair_confusion_matrix)
    np.save(args.out_leehon_confusion_matrix,leehon_confusion_matrix)

    print np.diag(leehon_confusion_matrix).sum()/leehon_confusion_matrix.sum()



     


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""
    calculate the models for every phone and output them
    along with a vector indicating the phone they belong to
    """)
    parser.add_argument('--phones',type=str,help='path to list of phones')
    parser.add_argument('--ncomponents',default=1,
                        type=int,
                        help='number of components in each mixture')
    parser.add_argument('--data_prefix',type=str,
                        help='prefix of files that contain the data')
    parser.add_argument('--data_suffix',type=str,
                        help='suffix for the data files')
    parser.add_argument('--model_list',type=str,
                        help='path containing the list of all the models')
    parser.add_argument('--out_confusion_matrix',type=str,
                        help='path of where to save the confusion matrix--which has the raw counts for the confusions')
    parser.add_argument('--out_known_pair_confusion_matrix',type=str,
                        help='path of where to save the confusion matrix--which has the raw counts for the confusions')
    parser.add_argument('--out_leehon_confusion_matrix',type=str,
                        help='path of where to save the confusion matrix--which has the leehon adjusted counts for the confusions')
    parser.add_argument('--leehon_mapping',
                        type=str,
                        default=None,
                        help='path to where the leehon mapping is contained')

    main(parser.parse_args())


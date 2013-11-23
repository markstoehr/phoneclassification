from __future__ import division
import numpy as np
import argparse
from CRunSVM import get_leehon_matrix

def main(args):
    """
    """

    phn_ids = [ l.split()[1] for l in open(args.phn_ids,'r').read().strip().split('\n')]
    leehon_mapping_dict = dict(tuple( l.split() for l in open(args.leehon_mapping,'r').read().strip().split('\n')))


    
    leehon_matrix = get_leehon_matrix(
        leehon_mapping_dict,
        phn_ids)

    num_models = len(args.templates)
    template0 = np.load(args.templates[0])
    template_shape = template0.shape[1:]
    log_odds = []
    constants = []
    class_ids = []
    component_ids = []
    for t_id, tpath in enumerate(args.templates):
        class_templates = np.load(tpath)
        class_templates = class_templates.reshape(
            class_templates.shape[0],
            np.prod(class_templates.shape[1:]))
        log_inv_class_templates = np.log(1-class_templates)
        class_log_odds = np.log(class_templates) - log_inv_class_templates
        class_sums = log_inv_class_templates.sum(-1)
        
        log_odds.extend(class_log_odds)
        constants.extend(class_sums)
        class_ids.extend( len(class_log_odds) * [t_id])
        component_ids.extend( np.arange(len(class_log_odds)))

    log_odds = np.array(log_odds).T
    constants = np.array(constants)
    class_ids = np.array(class_ids)
    component_ids = np.array(component_ids)

    confusion_matrix = np.zeros((num_models,
                                 num_models))

    all_scores = []
    all_labels= []
    all_components = []
    all_predicted_labels = []
    all_top_five_labels = []
    all_top_five_components = []
    num_data = 0
    num_true = 0
    for data_id, data_path in enumerate(args.data):
        print data_id,data_path
        data = np.load(data_path)
        data = data.reshape(len(data),np.prod(data.shape[1:]))
        scores = np.dot(data,log_odds) + constants
        scores_for_class = scores[:,np.arange(len(class_ids))[class_ids==data_id]]
        true_components = component_ids[class_ids==data_id][np.argmax(
            scores_for_class,1)]
        
        max_ids = scores.argmax(1)
        sorted_classes = class_ids[np.argsort(-scores,1).ravel()].reshape(scores.shape)
        sorted_components = component_ids[np.argsort(-scores,1).ravel()].reshape(scores.shape)

        num_data += len(sorted_classes)
        num_true += np.sum(class_ids[max_ids] == data_id)
        all_scores.extend(scores)
        all_labels.extend(len(scores) * [data_id])
        all_components.extend(true_components)
        all_predicted_labels.extend(class_ids[max_ids])
        all_top_five_labels.extend(sorted_classes[:,:5])
        all_top_five_components.extend(sorted_components[:,:5])

        confusions = np.bincount(class_ids[max_ids],minlength=num_models)
        confusion_matrix[data_id] += confusions
        assert data_id in np.argsort(confusion_matrix[data_id])[-5:]
        print num_true/num_data

    leehon_confusion_matrix = np.dot(leehon_matrix.T,
                                     np.dot(confusion_matrix,
                                            leehon_matrix))        
    np.save(args.out_confusion_matrix,confusion_matrix)
    np.save(args.out_leehon_confusion_matrix,leehon_confusion_matrix)

    print np.diag(leehon_confusion_matrix).sum()/leehon_confusion_matrix.sum()


    print num_true/num_data
    np.save(args.out_scores,scores)
    np.save(args.out_labels,all_labels)
    np.save(args.out_components,all_components)
    np.save(args.out_top_predicted_labels,all_top_five_labels)
    np.save(args.out_top_predicted_components,all_top_five_components)


if __name__=="__main__":
    parser = argparse.ArgumentParser("""
    Testing the Bernoulli likelihood ratio tests
    """)
    parser.add_argument('--data',
                        type=str,
                        nargs='+',
                        help='paths to where the data are kept')
    parser.add_argument('--templates',
                        type=str,
                        nargs='+',
                        help='paths to where the templates are saved, templates should correspond to the models associated with the data and they should be in the same order')
    parser.add_argument('--out_scores',
                        type=str,
                        help='path to save the scores to')
    parser.add_argument('--out_labels',
                        type=str,
                        help='path to save the labels to')
    parser.add_argument('--out_components',
                        type=str,
                        help='path to save the component with the largest score under the true model likelihood')
    parser.add_argument('--out_top_predicted_labels',
                        type=str,
                        help='path to save the top predicted labels to')
    parser.add_argument('--out_top_predicted_components',
                        type=str,
                        help='path to save the top predicted components to')

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

    main(parser.parse_args())

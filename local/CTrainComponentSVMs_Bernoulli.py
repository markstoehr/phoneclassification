from __future__ import division
import numpy as np
import argparse, itertools
from sklearn import svm
from template_speech_rec import configParserWrapper
from TestSVMBernoulli import get_bernoulli_templates

def get_false_predicted_label_ranks(
                        label,component,
                        true_predicted_label_ranks,
                        predicted_labels,
                        predicted_components):
    """
    The true_predicted_label_ranks, predicted_labels, predicted_components
    are assumed to only contain those entries which correspond to
    the labels being a certain phone with idea phn_id != label
    """
    false_label_ranks = np.argmax(
        (predicted_labels == label) * (predicted_components == component),1).astype(int)
    has_false_label = np.max(
        (predicted_labels == label) * (predicted_components == component),1).astype(int)
    false_label_ranks[-has_false_label] = 1+false_label_ranks.max()
    return false_label_ranks


def main(args):
    """
    For each label and component constructed a positive and negative
    training set and train a linear SVM to separate them
    """
    log_odds, constants, template_phn_ids, template_component_ids = get_bernoulli_templates(args.templates)

    true_examples = []
    false_examples = []
    for phn_id, fl in enumerate(args.data):
        X = np.load(fl)
        X_shape = X.shape[1:]
        X = X.reshape(X.shape[0],np.prod(X_shape))

        bernoulli_scores = np.dot(X,log_odds) + constants
        sorted_bernoulli_phn_ids = template_phn_ids[np.argsort(-bernoulli_scores,1).ravel()].reshape(bernoulli_scores.shape)
        sorted_bernoulli_component_ids = template_component_ids[np.argsort(-bernoulli_scores,1).ravel()].reshape(bernoulli_scores.shape)

        component_indices = np.argmax((sorted_bernoulli_component_ids == args.component_id) * (sorted_bernoulli_phn_ids == args.label),1)
        best_component_indices = np.argmax(sorted_bernoulli_phn_ids==args.label,1)



        if phn_id == args.label:
            # consider a true example if it is in the top 6
            # and the component is the right component

            true_examples.extend(
                X[ (component_indices==best_component_indices) *
                   (component_indices < 10)])
        else:
            # we find the index for the true phone
            true_phn_idx = np.argmin(((sorted_bernoulli_phn_ids != phn_id).astype(int)*log_odds.shape[1] + 1 ) * (1+np.arange(log_odds.shape[1])),1)

            false_examples.extend(
            X[ # (component_indices < true_phn_idx)*
               (component_indices < 6) ])


    if min(len(true_examples),len(false_examples)) < 20:
        print "len(true_examples)=%d,len(false_examples)=%d" % (len(true_examples),len(false_examples))
        print "So no svm trained for label %d component %d" % (args.label,args.component_id)
        return

    y = np.array(len(true_examples) * [0] + len(false_examples)*[1])
    X = np.array(true_examples + false_examples)
    del true_examples
    del false_examples



    config_d = configParserWrapper.load_settings(open(args.config,'r'))

    penalty_names = config_d['SVM']['penalty_list'][::2]
    penalty_values = tuple( float(k) for k in config_d['SVM']['penalty_list'][1::2])

    for penalty_name, penalty_value in itertools.izip(penalty_names,penalty_values):
        if args.v:
            print '%s %s' % ('linear', penalty_name)

        if config_d['SVM']['kernel'] == 'linear':
            clf = svm.LinearSVC(C=penalty_value,
                                loss='l1')
            clf.fit(X,y)
        else:
            import pdb; pdb.set_trace()



        coef = clf.coef_.reshape(clf.coef_.size)
        y_hat = np.dot(X,coef) + clf.intercept_[0]

        print np.sum((y_hat>0).astype(int) == y.astype(int))/len(y)
        np.save('%s_%s_%s_coef.npy' % (args.out_svm_prefix,config_d['SVM']['kernel'],
                                           penalty_name),
                    coef)
        np.save('%s_%s_%s_intercept.npy' % (args.out_svm_prefix,config_d['SVM']['kernel'],
                                           penalty_name),
                    clf.intercept_)




if __name__=="__main__":
    parser = argparse.ArgumentParser("""For each component and model
    we construct a positive and negative data subset and then
    train an SVM""")
    parser.add_argument('-v',
                        action='store_true',
                        help='verbosity flag')

    parser.add_argument('--config',
                        type=str,
                        default='conf/main.config',
                        help='configuration file')

    parser.add_argument('--label',
                        type=int,
                        help='integer id indicating the label for the svm we are training')
    parser.add_argument('--component_id',
                        type=int,
                        help='integer id indicating the mixture component for the given label we are training')
    parser.add_argument('--data',
                        type=str,
                        nargs='+',
                        help='paths to where the data are kept, in order of phone ids')
    parser.add_argument('--templates',
                        type=str,
                        nargs='+',
                        help='list of templates to use for classification')
    # parser.add_argument('--lengths',
    #                     type=str,
    #                     nargs='+',
    #                     help='paths to where the length files are kept, in order of phone ids')
    parser.add_argument('--labels',
                        type=str,
                        help='path to the true labels'
    )
    parser.add_argument('--components',
                        type=str,
                        help='path to the true components'
    )
    parser.add_argument('--predicted_labels',
                        type=str,
                        help='path to the file containing the predicted labels')
    parser.add_argument('--predicted_components',
                        type=str,
                        help='path to the file containing the predicted components')
    parser.add_argument('--out_svm_prefix',
                        type=str,
                        help='prefix to the file path where the svm coefficient and intercepts are going to be saved--they will also be indexed by label id and component id')
    main(parser.parse_args())


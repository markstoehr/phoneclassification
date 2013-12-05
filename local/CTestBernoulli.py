from __future__ import division
import numpy as np
import argparse,itertools
from CRunSVM import get_leehon_matrix
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

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
        if t_id > 0:
            if class_templates.shape[1] != log_odds[0].size:
                import pdb; pdb.set_trace()
        log_inv_class_templates = np.log(1-class_templates)
        class_log_odds = np.log(class_templates) - log_inv_class_templates
        class_sums = log_inv_class_templates.sum(-1)
        
        log_odds.extend(class_log_odds)
        constants.extend(class_sums)
        class_ids.extend( len(class_log_odds) * [t_id])
        component_ids.extend( np.arange(len(class_log_odds)))

    log_odds = np.array(log_odds).T
    constants = np.array(constants)
    class_diffs = np.zeros(log_odds.T.shape)
    class_counts = np.zeros(constants.shape)
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
        data_shape = data.shape[1:]
        data = data.reshape(len(data),np.prod(data.shape[1:]))
        scores = np.dot(data,log_odds) + constants

        scores_for_class = scores[:,np.arange(len(class_ids))[class_ids==data_id]]

        true_components = component_ids[class_ids==data_id][np.argmax(
            scores_for_class,1)]
        
        # get the mistakes
        scores_for_not_class = scores[:,np.arange(len(class_ids))[class_ids !=data_id]]
        best_false_components = component_ids[class_ids!=data_id][np.argmax(
            scores_for_not_class,1)]
        best_false_class = class_ids[class_ids!=data_id][np.argmax(scores_for_not_class,1)]

        # diff = np.zeros(data.shape[1:])
        # n = len(data)
        # for example_id,example in enumerate(data):
        #     try:
        #         diff +=(
        #         example* ( log_odds.T[ (class_ids ==data_id) *
        #                              (component_ids == true_components[example_id])].reshape(diff.shape)
        #                    - log_odds.T[ (class_ids ==best_false_class[example_id]) *
        #                                (component_ids == best_false_components[example_id])].reshape(diff.shape)) + ( constants[ (class_ids ==data_id) *
        #                                                                                                                          (component_ids == true_components[example_id])].reshape(1)
        #                                                                                                               - constants[ (class_ids ==best_false_class[example_id]) *
        #                                                                                                                            (component_ids == best_false_components[example_id])].reshape(1)))
        #     except:
        #         import pdb; pdb.set_trace()

        # diff = (1./n * diff).reshape(data_shape)
        # plt.close('all')

        # fig = plt.figure(1, (10,10))
        # grid = ImageGrid(fig, 111, # similar to subplot(111)
        #                  nrows_ncols = (4,2), # creates 2x2 grid of axes
        #          axes_pad=0.001, # pad between axes in inch.
        #              )

        # for j in xrange(8):
        #     try:
        #         grid[j].imshow(diff[:,:,j].T,cmap="hot",origin='lower')

        #         grid[j].spines['bottom'].set_color('red')
        #         grid[j].spines['top'].set_color('red')
        #         grid[j].spines['left'].set_color('red')
        #         grid[j].spines['right'].set_color('red')
        #         for a in grid[j].axis.values():
        #             a.toggle(all=False)
        #     except:
        #         import pdb; pdb.set_trace()
                
        # plt.savefig('plots/class_%d_diff_vis.png' % data_id,bbox_inches="tight")


        # np.save('exp/diff_%d.npy' % data_id,diff.reshape(data_shape))


        max_ids = scores.argmax(1)
        sorted_classes = class_ids[np.argsort(-scores,1).ravel()].reshape(scores.shape)
        sorted_components = component_ids[np.argsort(-scores,1).ravel()].reshape(scores.shape)

        # diff = np.zeros(data.shape[1:])
        # too_low_class_diff = np.zeros((np.sum(class_ids==data_id),) + data.shape[1:])

        # n = 0
        # for example_id,example in enumerate(data):
        #     try:
        #         if class_ids[max_ids][example_id] == data_id: continue
        #         n+=1
        #         cur_diff =(
        #         example* ( log_odds.T[ (class_ids ==data_id) *
        #                              (component_ids == true_components[example_id])].reshape(diff.shape)
        #                    - log_odds.T[ (class_ids ==best_false_class[example_id]) *
        #                                (component_ids == best_false_components[example_id])].reshape(diff.shape)) + ( constants[ (class_ids ==data_id) *
        #                                                                                                                          (component_ids == true_components[example_id])].reshape(1)
        #                                                                                                               - constants[ (class_ids ==best_false_class[example_id]) *
        #                                                                                                                            (component_ids == best_false_components[example_id])].reshape(1)))
        #         diff += cur_diff
        #         true_id = true_components[example_id]
        #         too_low_class_diff[true_id] += cur_diff
        #         false_id = np.argmax((class_ids ==best_false_class[example_id]) *
        #                                                                                                                            (component_ids == best_false_components[example_id]))
        #         class_diffs[false_id] -= cur_diff
        #         class_counts[false_id] += 1
        #     except:
        #         import pdb; pdb.set_trace()

        # diff = (1./n * diff).reshape(data_shape)
        # too_low_class_diff = (1./n * too_low_class_diff).reshape( (len(too_low_class_diff),) + data_shape)
        # plt.close('all')

        # fig = plt.figure(1, (10,10))
        # grid = ImageGrid(fig, 111, # similar to subplot(111)
        #                  nrows_ncols = (4,2), # creates 2x2 grid of axes
        #          axes_pad=0.001, # pad between axes in inch.
        #              )

        # for j in xrange(8):
        #     try:
        #         grid[j].imshow(diff[:,:,j].T,cmap="hot",origin='lower',vmin=diff.min(),vmax=diff.max())

        #         grid[j].spines['bottom'].set_color('red')
        #         grid[j].spines['top'].set_color('red')
        #         grid[j].spines['left'].set_color('red')
        #         grid[j].spines['right'].set_color('red')
        #         for a in grid[j].axis.values():
        #             a.toggle(all=False)
        #     except:
        #         import pdb; pdb.set_trace()
                
        # plt.savefig('plots/class_%d_diff_vis_mistakes.png' % data_id,bbox_inches="tight")


        # np.save('exp/diff_%d_mistakes.npy' % data_id,diff.reshape(data_shape))
        
        # for i in xrange(len(too_low_class_diff)):
        #     plt.close('all')
            
        #     fig = plt.figure(1, (10,10))
        #     grid = ImageGrid(fig, 111, # similar to subplot(111)
        #                      nrows_ncols = (4,2), # creates 2x2 grid of axes
        #                      axes_pad=0.001, # pad between axes in inch.
        #              )

        #     for j in xrange(8):
        #         try:
        #             grid[j].imshow(too_low_class_diff[i][:,:,j].T,cmap="hot",origin='lower',vmin=too_low_class_diff[i].min(),vmax=too_low_class_diff[i].max())
                    
        #             grid[j].spines['bottom'].set_color('red')
        #             grid[j].spines['top'].set_color('red')
        #             grid[j].spines['left'].set_color('red')
        #             grid[j].spines['right'].set_color('red')
        #             for a in grid[j].axis.values():
        #                 a.toggle(all=False)
        #         except:
        #             import pdb; pdb.set_trace()
                
        #     plt.savefig('plots/class_%d_diff_vis_mistakes_too_low_%d.png' % (data_id,i),bbox_inches="tight")


        #     np.save('exp/diff_%d_mistakes_too_low_%d.npy' % (data_id,i),too_low_class_diff[i].reshape(data_shape))

        
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


    # for class_component_id, (cdiff,diff_count) in enumerate(itertools.izip(class_diffs,class_counts)):
    #     class_id = class_ids[class_component_id]
    #     component_id = component_ids[class_component_id]
    #     print class_id, component_id
    #     diff = cdiff.reshape(data_shape)/diff_count
    #     plt.close('all')

    #     fig = plt.figure(1, (10,10))
    #     grid = ImageGrid(fig, 111, # similar to subplot(111)
    #                      nrows_ncols = (4,2), # creates 2x2 grid of axes
    #              axes_pad=0.001, # pad between axes in inch.
    #                  )

    #     for j in xrange(8):
    #         try:
    #             grid[j].imshow(diff[:,:,j].T,cmap="hot",origin='lower',vmin=diff.min(),vmax=diff.max())

    #             grid[j].spines['bottom'].set_color('red')
    #             grid[j].spines['top'].set_color('red')
    #             grid[j].spines['left'].set_color('red')
    #             grid[j].spines['right'].set_color('red')
    #             for a in grid[j].axis.values():
    #                 a.toggle(all=False)
    #         except:
    #             import pdb; pdb.set_trace()
                
    #     plt.savefig('plots/class_%d_diff_vis_mistakes_%d.png' % (class_id,component_id),bbox_inches="tight")


    #     np.save('exp/diff_%d_mistakes_%d.npy' % (class_id,component_id),diff.reshape(data_shape))


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

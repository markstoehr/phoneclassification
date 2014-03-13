from __future__ import division
import numpy as np
import argparse, itertools
from sklearn import svm
from template_speech_rec import configParserWrapper
from TestSVMBernoulli import get_bernoulli_templates
from scipy.ndimage.filters import maximum_filter
from amitgroup.stats import bernoullimm
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.cm as cm

def get_maximal_patches(X,S,patch_radius=2):
    """
    """
    k = 2*patch_radius+1
    edge_sub_shape=(k,k,8)
    edge_view_shape=tuple(np.subtract(X.shape,edge_sub_shape)+1)[:-1]+edge_sub_shape
    edge_arr_view= np.lib.stride_tricks.as_strided(X,edge_view_shape,X.strides[:-1] + X.strides )
    edge_arr_sums = edge_arr_view.sum(-1).sum(-1).sum(-1)
    edge_local_maxes = maximum_filter(edge_arr_sums,size=(k,k),cval=0,mode='constant')
    local_max_patches = edge_arr_view[edge_arr_sums >= edge_local_maxes]

    spec_sub_shape = (k,k)

    spec_view_shape=tuple(np.subtract(S.shape,spec_sub_shape)+1)+spec_sub_shape
    spec_arr_view=np.lib.stride_tricks.as_strided(S,spec_view_shape,S.strides *2 )

    return local_max_patches, spec_arr_view[edge_arr_sums >= edge_local_maxes]



def main(args):
    """
    For each label and component constructed a positive and negative
    training set and train a linear SVM to separate them
    """
    config_d = configParserWrapper.load_settings(open(args.config,'r'))

    true_examples = []
    false_examples = []
    mean = 0
    total = 0
    num_less_than_eq = np.zeros(20)

    all_X_patches = []
    all_S_patches = []
    for phn_id, (fl_edge, fl_spec) in enumerate(itertools.izip(args.data,args.data_spec)):
        if len(all_X_patches) > 100000: break
        print phn_id

        X = np.load(fl_edge)
        X_shape = X.shape[1:]

        S = np.load(fl_spec)
        if args.do_exp_weighted_divergence:
            S *= np.exp(S)


        X_patches = []
        S_patches = []
        for i in xrange(len(X)):
            if len(X_patches) > 1000: break
            p_edge,p_spec = get_maximal_patches(X[i],S[i],patch_radius=args.patch_radius)
            X_patches.extend(p_edge)
            S_patches.extend(p_spec)

        num_new_patches = len(X_patches)
        print phn_id, num_new_patches
        all_X_patches.extend(X_patches)
        all_S_patches.extend(S_patches)


    X = np.array(all_X_patches)
    S = np.array(all_S_patches)
    data_shape = X.shape[1:]
    X = X.reshape(X.shape[0],np.prod(data_shape))
    bmm = bernoullimm.BernoulliMM(n_components=args.n_components,
                                  n_init= 20,
                                  n_iter= 500,
                                  random_state=0,
                                  verbose=args.v, tol=1e-6)
    bmm.fit(X)

    # check above 30
    use_means = bmm.predict_proba(X).sum(0) > 30
    print use_means.sum()
    try:
        np.save(args.save_parts,bmm.means_.reshape(*( (bmm.n_components,)+data_shape))[use_means])
    except:
        import pdb; pdb.set_trace()
    S_shape = S.shape[1:]
    S_clusters = bmm.cluster_underlying_data(S.reshape(len(S),np.prod(S_shape)),X).reshape(
            *( (bmm.n_components,) + S_shape))[use_means]
    np.save(args.spec_save_parts,S_clusters)

    ncols = int(np.sqrt(args.n_components))
    nrows = int(np.ceil(args.n_components/ncols))


    if args.viz_spec_parts is not None:
        plt.close('all')
        fig = plt.figure(1, (6, 6))
        grid = ImageGrid(fig, 111, # similar to subplot(111)
                             nrows_ncols = (nrows,ncols ), # creates 2x2 grid of axes
                             axes_pad=0.001, # pad between axes in inch.
                     )

        for i in xrange(S_clusters.shape[0]):

            try:
                grid[i].imshow(S_clusters[i],cmap=cm.binary,interpolation='nearest')
                grid[i].spines['bottom'].set_color('red')
                grid[i].spines['top'].set_color('red')
                grid[i].spines['left'].set_color('red')
                grid[i].spines['right'].set_color('red')
                for a in grid[i].axis.values():
                    a.toggle(all=False)
            except:
                import pdb; pdb.set_trace()

        for i in xrange(S_clusters.shape[0],nrows*ncols):
            try:
                grid[i].spines['bottom'].set_color('red')
            except: import pdb; pdb.set_trace()
            grid[i].spines['top'].set_color('red')
            grid[i].spines['left'].set_color('red')
            grid[i].spines['right'].set_color('red')

            for a in grid[i].axis.values():
                a.toggle(all=False)

        plt.savefig('%s' % args.viz_spec_parts
                                           ,bbox_inches='tight')











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

    parser.add_argument('--data',
                        type=str,
                        nargs='+',
                        help='paths to where the data are kept, in order of phone ids')

    parser.add_argument('--data_spec',
                        type=str,
                        nargs='+',
                        help='paths to where the spec data are kept, in order of phone ids')


    parser.add_argument('--save_parts',
                        type=str,
                        help='paths to where the data are kept, in order of phone ids')

    parser.add_argument('--spec_save_parts',
                        type=str,
                        help='paths to where the spec data are kept, in order of phone ids')
    parser.add_argument('--viz_spec_parts',
                        type=str,
                        default=None,
                        help='paths to where the spec data are kept, in order of phone ids')
    parser.add_argument('--n_components',
                        type=int,
                        default=50,
                        help='number of components')
    parser.add_argument('--patch_radius',
                        default=2,
                        type=int,
                        help='radius for the patch')
    parser.add_argument('--do_exp_weighted_divergence',
                        action='store_true',help='whether to do the exponentially-weighted differences -i.e. a KL divergence on the spectrogram for edges')
    # parser.add_argument('--lengths',
    #                     type=str,
    #                     nargs='+',
    #                     help='paths to where the length files are kept, in order of phone ids')
    main(parser.parse_args())


import numpy as np
import argparse, itertools
from template_speech_rec import configParserWrapper
from amitgroup.stats import bernoullimm
from template_speech_rec import configParserWrapper


def main(args):
    """
    get all the data matrices and process the data
    """
    config_d = configParserWrapper.load_settings(open(args.config,'r'))
    phones = np.loadtxt(args.phones,dtype=str)
    max_n_classifiers = len(phones) * args.ncomponents



    classifier_id = 0
    for phone_id, phone in enumerate(phones):
        if args.v:
            print "Working on phone %s which has id %d" % (phone,phone_id)
            print "classifier_id = %d" % classifier_id
        X = np.load('%s/%s_%s' % ( args.data_prefix,
                                   phone,
                                   args.data_suffix))


        
        if phone_id == 0:
            avgs = np.zeros((max_n_classifiers,
                               ) + X.shape[1:])
            counts = np.zeros(max_n_classifiers
                               )
            # will keep track of which average belongs to which
            # phone and mixture component--this allows us to
            # drop mixture components if they are potentially
            # not helping
            weights = np.zeros(max_n_classifiers,dtype=float)
            meta = np.zeros((max_n_classifiers
                             ,2),dtype=int)
            
        if args.ncomponents == 1:
            avgs[phone_id] = X.mean(0)
            counts[phone_id] = X.shape[0]
            weights[phone_id] = 1
            meta[phone_id,0] = phone_id
            meta[phone_id,1] = 0
            classifier_id += 1
        else:
            bmm = bernoullimm.BernoulliMM(n_components=args.ncomponents,
                                          n_init= config_d['EM']['n_init'],
                                          n_iter= config_d['EM']['n_iter'],
                                          tol=config_d['EM']['tol'],
                                          random_state=config_d['EM']['random_seed'],
                                          verbose=args.v)
            bmm.fit(X)

            responsibilities = bmm.predict_proba(X)
            component_counts = responsibilities.sum(0)
            no_use_components = component_counts < config_d['EM']['min_data_count']
            while no_use_components.sum() > 0:
                n_use_components = len(component_counts) -1
                bad_component = np.argmin(component_counts)
                use_components = np.ones(len(component_counts),
                                         dtype=bool)
                use_components[bad_component] = False
                bmm.means_ = bmm.means_[use_components]
                bmm.weights_ = bmm.weights_[use_components]
                bmm.weights_ /= bmm.weights_.sum()
                bmm.n_components = n_use_components
                bmm.log_odds_, bmm.log_inv_mean_sums_ = bernoullimm._compute_log_odds_inv_means_sums(bmm.means_)
                bmm.n_iter = 1
                bmm.init_params=''
                bmm.fit(X)
                responsibilities = bmm.predict_proba(X)
                component_counts = responsibilities.sum(0)
                no_use_components = component_counts < config_d['EM']['min_data_count']
                print component_counts

            n_use_components = bmm.n_components
            cur_means = bmm.means_.reshape(
                *((n_use_components,)
                  + avgs.shape[1:]))
            cur_counts = component_counts
            cur_weights = bmm.weights_
            avgs[classifier_id:classifier_id+
                 n_use_components] = cur_means
            counts[classifier_id:
                   classifier_id + n_use_components] = cur_counts
            weights[classifier_id:
                    classifier_id + n_use_components] = cur_weights
            meta[classifier_id:classifier_id+n_use_components,0] = phone_id
            meta[classifier_id:classifier_id+n_use_components,1] = np.arange(n_use_components)
            
            # make sure we move forward in the vector
            classifier_id += n_use_components
            

    print "Total of %d models" % classifier_id
    np.save('%s/avgs_%s' % (args.out_prefix, args.out_suffix),
            avgs[:classifier_id])
    np.save('%s/counts_%s' % (args.out_prefix, args.out_suffix),
            counts[:classifier_id])
    np.save('%s/weights_%s' % (args.out_prefix, args.out_suffix),
            weights[:classifier_id])

    np.save('%s/meta_%s' % (args.out_prefix, args.out_suffix),
            meta[:classifier_id])
     


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""
    calculate the models for every phone and output them
    along with a vector indicating the phone they belong to
    """)
    parser.add_argument('--phones',type=str,help='path to list of phones')
    parser.add_argument('--config',
                        type=str,
                        default='conf/main.config',
                        help='configuration file')

    parser.add_argument('--ncomponents',default=1,
                        type=int,
                        help='number of components in each mixture')
    parser.add_argument('--data_prefix',type=str,
                        help='prefix of files that contain the data')
    parser.add_argument('--data_suffix',type=str,
                        help='suffix for the data files')
    parser.add_argument('--out_prefix',type=str,
                        help='prefix to the file where the data will be saved')
    parser.add_argument('--out_suffix',type=str,
                        help='suffix to where the files will be saved')
    parser.add_argument('-v',action='store_true',help='whether to have verbose output')
    main(parser.parse_args())


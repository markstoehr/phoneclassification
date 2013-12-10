from __future__ import division
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.io import wavfile
import template_speech_rec.get_train_data as gtrd
from template_speech_rec import configParserWrapper
from amitgroup.features import code_parts, spread_patches, spread_patches_new
from stride_parts import code_spread_parts
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_spec_chunk_binary_features(S_chunk,P_chunk,savename,spec_means):

    plt.close('all')
    plt.subplot(211)
    plt.imshow(S_chunk.T,cmap="hot",origin='lower',interpolation='nearest')
    use_idx = P_chunk.T.ravel() >0
    x_coords = (np.arange(P_chunk.T.size) % P_chunk.T.shape[1])[use_idx]
    y_coords = (np.arange(P_chunk.T.size) / P_chunk.T.shape[1])[use_idx]
    plt.scatter(x_coords,y_coords,marker='+',c='b',lw=.25)
    plt.subplot(212)
    plt.imshow(spec_means.T,origin='lower',cmap='hot',interpolation='nearest')
    plt.savefig(savename,bbox_inches="tight")
    plt.close('all')


def main(args):
    """
    Visualize the parts and where they are active on 
    a given utterance
    """
    
    
    config_d = configParserWrapper.load_settings(open(args.c,'r'))

    part_means = np.load(args.part_edge_means)
    spec_means = np.load(args.spec_means)
    log_part_means = np.log(part_means)
    log_part_inv_means = np.log(1-part_means)
    log_part_odds = log_part_means - log_part_inv_means
    part_shape = log_part_odds.shape[1:]
    log_part_odds = log_part_odds.reshape(log_part_odds.shape[0],
                                          np.prod(part_shape)).T

    constants = log_part_inv_means.sum(-1).sum(-1).sum(-1)
    sr, x = wavfile.read(args.wavfile)
    S,sample_mapping, sample_to_frames = gtrd.get_spectrogram_use_config(x,config_d['SPECTROGRAM'],return_sample_mapping=True)
    E = gtrd.get_edge_features_use_config(S.T,config_d['EDGES'])
    

    X  = code_spread_parts(E,
                           log_part_odds,
                           constants,part_shape,part_shape[:-1],
                           count_threshold=20,likelihood_threshold=-300/200. * np.prod(part_shape))



    for part_id in xrange(X.shape[-1]):
        for i in xrange(int((E.shape[0] - 200)/50.)):
        
            P_chunk = X[i*50:i*50+200,:,part_id]
            S_chunk = S[i*50:i*50+200]
            savename = '%s_%d_%d.png' % (args.viz_save_prefix,
                                         part_id,
                                         i)
        
            plot_spec_chunk_binary_features(S_chunk,P_chunk,savename,spec_means[part_id])

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Visualize the 
    """)
    parser.add_argument('-c',type=str,default='conf/main.config',
                        help='config file')
    parser.add_argument('--part_edge_means',type=str,
                        help='path to the part means')
    parser.add_argument('--spec_means',type=str,
                        help='path to the part spec means')
    parser.add_argument('--wavfile',type=str,
                        help='path to the wavfile')
    parser.add_argument('--viz_save_prefix',
                        type=str,
                        help='path to save the visualized parts to')
    main(parser.parse_args())

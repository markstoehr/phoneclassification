#!/usr/bin/python

import numpy as np
import argparse, itertools
from template_speech_rec import configParserWrapper
from scipy.io import wavfile
import template_speech_rec.get_train_data as gtrd
import filterbank as fb    
from transforms import binary_phase_features, preemphasis, process_wav
import matplotlib.pyplot as plt

def main(args):
    """
    Get the start and end frame associated to each example
    and then extract the example and save into a matrix

    we make the assumption that all the examples have the exact
    same length as the first
    """
    phones = np.loadtxt(args.phones,dtype='str')
    config_d = configParserWrapper.load_settings(open(args.c,'r'))

    # uttid_dict = dict(tuple( (uttid,k) for k,uttid in enumerate(open(args.uttids,'r').read().strip().split('\n'))))

    examples = []
    examples_sfa = []

    get_dict = lambda x: dict(l.split() for l in open(x,'r').read().strip().split('\n'))
    transcript_dict = get_dict(args.transcripts)
    output_dict = get_dict(args.outputs)
    output_waves_dict = get_dict(args.output_waves)

    # we assume that Q should have at least length 2 for this
    # code to work

    htemp, dhtemp, ddhtemp, tttemp = fb.hermite_window(
        config_d['BPF']['winsize']-1,
        config_d['BPF']['order'],
        config_d['BPF']['half_time_support'])

    h = np.zeros((htemp.shape[0],
                  htemp.shape[1]+1))
    h[:,:-1] = htemp

    dh = np.zeros((dhtemp.shape[0],
                   dhtemp.shape[1]+1))
    dh[:,:-1] = dhtemp


    tt = (2*tttemp[-1] -tttemp[-2])*np.ones(tttemp.shape[0]+1)
    tt[:-1] = tttemp


    oversampling=config_d['BPF']['oversampling']
    T_s=config_d['OBJECT']['t_s']

    gsigma = config_d['BPF']['gsigma']
    gfilter= fb.get_gauss_filter(config_d['BPF']['gt'],
                                 config_d['BPF']['gf'],
                                 gsigma)
    run_transform = lambda x: binary_phase_features(
        preemphasis(process_wav(x),preemph=config_d['BPF']['preemph']),
        config_d['BPF']['sample_rate'],
        config_d['BPF']['freq_cutoff'],
        config_d['BPF']['winsize'],
        config_d['BPF']['nfft'],
        oversampling,
        h,
        dh,
        tt,
        gfilter,
        gsigma,
        config_d['BPF']['fthresh'],
        config_d['BPF']['othresh'],
        spread_length=config_d['BPF']['spread_length'],
        return_midpoints=True)


    for phone_id, phone in enumerate(phones):
        trans_list = transcript_list(transcript_dict[phone])
        Ss, TFs, example_length = extract_examples(trans_list, run_transform, T_s)
        if phone_id > 0:
            assert old_example_length == example_length
        else:
            old_example_length = example_length

            
        print output_dict[phone], output_waves_dict[phone], transcript_dict[phone]
        np.save(output_dict[phone],TFs)
        np.save(output_waves_dict[phone],Ss)
        print phone, example_length
                                                           
    

def transcript_list(transcript_path):
    """
    From a path to a transcript return a list of tuples where the
    first entry of the tuple is the path to the wav file
    and the second entry of the tuple is a list of (start_sample, end_sample) pairs
    """
    trans_list = []
    for line_id, line in enumerate(open(transcript_path,'r')):
        split_line = line.strip().split()
        fpath = split_line[0]
        split_line = split_line[1:]
        sample_starts = np.array(tuple(int(k) for k in split_line[::3]), dtype=int)
        sample_ends = np.array( tuple( int(k) for k in split_line[1::3]),dtype=int)
        trans_list.append((fpath,
                           zip(sample_starts,sample_ends)))

    return trans_list

def extract_examples(trans_list,
                        run_transform,
                        T_s):
    """
    Parameters
    ----------
    trans_list : list
        List of every wav file along with the start and end samples
        for each example

    run_transform : function
        returns the wavelet spectrogram frequence average along 
        with the midpoints for each frame

    T_s : int
        Number of samples expected to be within an example
    """
    S, TF, midpoints = run_transform(wavfile.read(trans_list[0][0])[1])

    # we assume that the difference between subsequent midpoints is
    # uniform so that
    example_window_radius = int((T_s/np.diff(midpoints)[0])/2)
    n_examples = sum( len(t[1]) for t in trans_list)
    example_length  = 1+2*example_window_radius
    example_shape = (example_length,) + TF.shape[1:]
    print example_length
    Ss = np.zeros((n_examples,example_length) + S.shape[1:])
    TFs = np.zeros((n_examples,)+example_shape)
    cur_example = 0
    for t_id, t in enumerate(trans_list):
        if t_id % 100 == 0:
            print t_id, t, cur_example
        S,TF, midpoints = run_transform(wavfile.read(t[0])[1])
        N  = TF.shape[0]
        for example in t[1]:
            mid = (example[1] + example[0])/2
            mid_window = np.argmin(np.abs(midpoints - mid))
            example_indices = np.arange(example_length)+ mid_window - example_window_radius
            # symmetrize via mirroring
            example_indices[example_indices < 0] *= -1
            example_indices[example_indices > N-1] = N-1- (example_indices[example_indices > N-1] - N+1)
            TFs[cur_example] = TF[example_indices]
            Ss[cur_example] = S[example_indices]
            cur_example += 1

    return Ss,TFs, example_length


if __name__=="__main__":
    parser = argparse.ArgumentParser("""
    Compute the frame length for the examples (based on the number
    of samples for an example length)
    and then extract the best centered example for each of the 
    data.
    """)
    parser.add_argument('-c',
                        type=str,
                        default='conf/main.config',
                        help='configuration file should contain a section about spectrogram computation')
    parser.add_argument('--uttids',
                        type=str,
                        default='train.uttids',
                        help='utterance ids so that we can map back to integers')
    parser.add_argument('--transcripts',
                        type=str,
                        default='aa_train.frame_trans',
                        help='list of paths to files containing the transcripts')
    parser.add_argument('--outputs',
                        type=str,
                        default='aa_train_examples.npy',
                        help='list of file paths to output the data to, each line should have two entries--a phone and a path for saving the matrix of examples of the phone or phone sequence')
                                 
    parser.add_argument('--output_waves',
                        type=str,
                        default=None,
                        help='If not None then this is the data matrix corresponding to the wavelet output same as --outputs but for wavelet tfrs')
    parser.add_argument('--phones',
                        type=str,
                        default='data/local/data/phone.list',
                        help='list of phones--this is useful for organizing the output')
    main(parser.parse_args())

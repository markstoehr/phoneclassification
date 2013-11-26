#!/usr/bin/python

import sys,argparse

def main(args):
    """
    Go through each line of the input file and find those lines which match the output uttids
    """

    test_scp= open(args.i,'r')

    uttids_fpaths = tuple(line.strip().split()
                             for line in test_scp.read().strip().split('\n'))

    test_scp.close()

    spks = set(open(args.spk,'r').read().strip().split('\n'))

    

    for uttid, fpath in uttids_fpaths:
        dr, spk, sentence = uttid.split('_')
        if spk in spks:
            print '%s %s' % (uttid, fpath)



    
if __name__=="__main__":
    parser = argparse.ArgumentParser("""
    Tool to split the utterance ids into training and development
    sets
    """)
    parser.add_argument('-i',type=str,
                        default='test_wav.scp',
                        help='script file containing an almagation of training and dev utterances')
    parser.add_argument('--spk',type=str,
                        default='conf/dev_spk',
                        help='utterance ids for the training or dev set')
    main(parser.parse_args())
    
    

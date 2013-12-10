from __future__ import division
import argparse,itertools

def main(args):
    phns2ids = dict(tuple( l.split() for l in open(args.phns2ids,'r').read().strip().split('\n')))
    leehon_mapping = dict(tuple( l.split() for l in open(args.leehon_mapping,'r').read().strip().split('\n')))

    leehon_mapping_ids = dict(
        (phns2ids[p],
         phns2ids[leehon_mapping[p]])
         for p in phns2ids.keys())

    predicted_labels = open(args.predicted_labels,'r').read().strip().split('\n')
    true_labels = [ l.split()[0] for l in open(args.true_labels,'r')]


    num_mistakes = 0
    for p,t in itertools.izip(predicted_labels,true_labels):
        if leehon_mapping_ids[p] != leehon_mapping_ids[t]:
            num_mistakes += 1

    if len(true_labels) != len(predicted_labels): import pdb; pdb.set_trace
    print 1-num_mistakes/len(true_labels)


if __name__=="__main__":
    parser = argparse.ArgumentParser("""
    Score the output from liblinear
    """)

    parser.add_argument('--phns2ids',type=str,
                        help='mapping from phones to ids')
    parser.add_argument('--leehon_mapping',type=str,
                        help='mapping from the 48 phones to 39 as described in the lee and hon paper')
    parser.add_argument('--predicted_labels',type=str,
                        help='predicted labels using the predict function from liblinear, assumed to just be a column of the predicted labels')
    parser.add_argument('--true_labels',type=str,
                        help='single column of the true labels')

    main(parser.parse_args())

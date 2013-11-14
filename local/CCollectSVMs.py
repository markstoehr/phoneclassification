import numpy as np
import argparse, itertools

def main(args):
    """
    """
    neg_labels = args.label_pairs[::2]
    pos_labels = args.label_pairs[1::2]
    single_labels = list(np.sort(np.unique(neg_labels+pos_labels)))

    label_indices = []
    for label_idx, (neg_label, pos_label) in enumerate(itertools.izip(neg_labels,pos_labels)):
        label_indices.append(
            (label_idx, (single_labels.index(neg_label),
             single_labels.index(pos_label))))

    label_indices = sorted(label_indices, key = lambda x: x[1])
    coefs = ()
    coefs_shape = None
    intercepts = ()
    out_ids = ()
    for idx, (neg_label_idx, pos_label_idx) in label_indices:
        out_ids +=        (  "%d %d %d %s %s" % (idx,
            neg_label_idx,
            pos_label_idx,
            args.coefs[idx],
            args.intercepts[idx]),)
        if coefs_shape is None:
            c = np.load(args.coefs[idx])

            if len(c.shape) == 2:
                c = c[0]

            coefs += (c,)
        else:
            c = np.load(args.coefs[idx])

            if len(c.shape) == 2:
                c = c[0]

            try: coefs += (c,)
            except: import pdb; pdb.set_trace()

        intercepts += (np.load(args.intercepts[idx])[0],)

    coefs = np.array(coefs)
    intercepts = np.array(intercepts)

    open(args.phns_to_ids,'w').write(
        '\n'.join( tuple("%d %s" % (k,p) for k,p in enumerate(single_labels))))
    open(args.out_ids,'w').write('\n'.join(out_ids))
    np.save(args.out_coefs,coefs)
    np.save(args.out_intercepts,intercepts)





if __name__=="__main__":
     parser = argparse.ArgumentParser("""
     Collect svms together: intercepts and coefficient vectors
     from a list""")
     parser.add_argument('--coefs',
                         type=str,
                         nargs='+',
                         help='paths to the coefficient vectors in order to the matching puts to --intercepts')
     parser.add_argument('--intercepts',
                         type=str,
                         nargs='+',
                         help='paths to the intercepts in order to the matching puts to --coefs')
     parser.add_argument('--out_coefs',type=str,help='path to where the out coefficient vector is going to be saved')
     parser.add_argument('--out_intercepts',type=str,
                         help='the intercepts in one vector')
     parser.add_argument('--label_pairs',type=str,nargs='+',help='list of the label strings as pairs for each of the svm coefficient vectors')
     parser.add_argument('--phns_to_ids',type=str,
                         help='path to file where the sorted phone list is saved for future reference')
     parser.add_argument('--out_ids',
                         type=str,
                         help='text file where the first column is an index, the second column is the label string, the third path to the coefficient file, and the fourth is the path the intercept file')
     main(parser.parse_args())

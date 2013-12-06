import numpy as np
import argparse
import scipy

def main(args):
    """
    Read through a directory looking for the pieces of a
    file for the kernel matrix generated in the file
    local/get_scat_features.m
    and write out the kernel to a file
    the pieces, in the correct order should be saved to a text file
    this file will read in the text file line by line
    get the labels and get the pieces of the kernel
    matrix and output a data file that is appropriate for
    liblinear
    """
    out_fl_handle = open(args.out_fl,'w')
    phns2ids = dict(tuple( l.split() for l in open(args.phns2ids,'r').read().strip().split('\n')))
    for fl_id, fl in enumerate(open(args.data_files,'r')):
        label, fname = fl.strip().split()
        phn_id = phns2ids[label]

        if args.ftype == 'ascii':
            data = np.loadtxt(fname)
        elif args.ftype == 'npy':
            data = np.load(fname)

        if args.label_files is None:
            try:
                labels = int(phn_id)* np.ones(len(data))
            except: import pdb; pdb.set_trace()
        else:
            pass

        for obs_id, obs in enumerate(data):
            out_fl_handle.write('%d ' % labels[obs_id])
            out_fl_handle.write( ' '.join(
                '%d:%s' % (i+1,f) for i,f in enumerate(obs.ravel())))
            out_fl_handle.write('\n')



    out_fl_handle.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser("""    Read through a directory looking for the pieces of a
    file for the kernel matrix generated in the file
    local/get_scat_features.m
    and write out the kernel to a file
    the pieces, in the correct order should be saved to a text file
    this file will read in the text file line by line
    get the labels and get the pieces of the kernel
    matrix and output a data file that is appropriate for
    liblinear

    each chunk should represent a different class
    otherwise the classes are read in from
    args.kernel_label_files
""")
    parser.add_argument('--data_files',
                        type=str,
                        help='path to the text file listing the kernel matrix files')
    parser.add_argument('--label_files',
                        type=str,
                        default=None,
                        help='path to the text file listing the labels for the rows of the kernel matrix')
    parser.add_argument('--ftype',
                        type=str,
                        default='ascii',
                        help='what type the data files are')
    parser.add_argument('--out_fl',
                        type=str,
                        help='file to save the collected outputs to')
    parser.add_argument('--phns2ids',type=str,
                        help='file containing two columns where the first is the list of phones and the second is the index corresponding to that phone label')
    main(parser.parse_args())

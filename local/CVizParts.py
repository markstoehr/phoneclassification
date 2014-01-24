import numpy as np
import argparse
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

def main(args):
    """
    Load in the S clusters and the parts
    """
    



    nrows=3
    ncols=3
        
    edge_means = np.load(args.edge_means)
    spec_means = np.load(args.spec_means)
        
    for i in xrange(edge_means.shape[0]):
        plt.close('all')
        fig = plt.figure(1, (6, 6))
        grid = ImageGrid(fig, 111, # similar to subplot(111)
                             nrows_ncols = (nrows,ncols ), # creates 2x2 grid of axes
                             axes_pad=0.001, # pad between axes in inch.
        )
        grid[0].imshow(spec_means[i].T,origin='lower',cmap='binary',interpolation='nearest')
        grid[0].spines['bottom'].set_color('green')
        grid[0].spines['top'].set_color('green')
        grid[0].spines['left'].set_color('green')
        grid[0].spines['right'].set_color('green')
        for a in grid[0].axis.values():
            a.toggle(all=False)

        for j in xrange(1,9):
            grid[j].imshow(edge_means[i][:,:,j-1].T,origin='lower',cmap='hot',interpolation='nearest',vmin=0,vmax=1)
            grid[j].spines['bottom'].set_color('green')
            grid[j].spines['top'].set_color('green')
            grid[j].spines['left'].set_color('green')
            grid[j].spines['right'].set_color('green')
            for a in grid[j].axis.values():
                a.toggle(all=False)


        plt.savefig('%s_%d.png' % (args.viz_save_prefix,i)
                                           ,bbox_inches='tight')

if __name__ =="__main__":
    parser = argparse.ArgumentParser("""
    Visualize the parts where each part gets its own file
    the top left part in the 3 by 3 grid is the mean over the 
    spectrogram clusters and the rest of the parts are formed 
    from the edges.
    """)
    parser.add_argument(
        '--viz_save_prefix',
        type=str,
        help="prefix for where to save the part files to, they will be appended with an index indicating the part number (so omit the .png)")
    parser.add_argument('--spec_means',
                        type=str,
                        help='path to the file containing the spectrogram means')
    parser.add_argument('--edge_means',
                        type=str,
                        help='path to the file containing the means for the edge features')
    main(parser.parse_args())

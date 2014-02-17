import numpy as np
import argparse
from phoneclassification.stride_parts import code_spread_parts

def main(args):
    E = np.load(args.data)
    part_means = np.load(args.parts)
    log_part_means = np.log(part_means)
    log_part_inv_means = np.log(1-part_means)
    log_part_odds = log_part_means - log_part_inv_means
    part_shape = log_part_odds.shape[1:]
    log_part_odds = log_part_odds.reshape(log_part_odds.shape[0],
                                          np.prod(part_shape)).T

    constants = log_part_inv_means.sum(-1).sum(-1).sum(-1)
    parts_data = []
    for E_example in E:
        parts_data.append(
            code_spread_parts(E_example,
                              log_part_odds,
                              constants,part_shape,part_shape[:-1],
                            count_threshold=30/200.*np.prod(part_shape),likelihood_threshold=-300/200. * np.prod(part_shape))[np.newaxis]
        )
    
    X = np.vstack(parts_data)
    np.save(args.out,X)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Get the parts representation for data
    """)
    parser.add_argument('--data',type=str,help='path to where the edge data is that will be converted to a parts representation')
    parser.add_argument('--parts',type=str,help='path to where the parts are stored')
    parser.add_argument('--out',type=str,help='path to where the new data with a parts representation will be stored')
    main(parser.parse_args())


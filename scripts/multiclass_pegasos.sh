datadir=`pwd`/data/local/data
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf
old_exp=`pwd`/exp/pairwise_bernoulli_thresh
exp=`pwd`/exp/multiclass_pegasos
scripts=`pwd`/scripts
plots=$exp/plots
mkdir -p $exp
mkdir -p $conf
mkdir -p $plots

# now we run the multiclass pegasos on the scattering features

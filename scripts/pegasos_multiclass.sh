datadir=`pwd`/data/local/data
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf
old_exp=`pwd`/exp/pairwise_bernoulli_thresh
exp=`pwd`/exp/pegasos
mkdir -p $exp
mkdir -p $conf

# estimate the set of models

python $local/estimate_all_models.py --phones $datadir/phone.list\
    --data_prefix $datadir\
    --data_suffix train_examples.npy\
    --out_prefix $old_exp\
    --out_suffix 1C.npy\
    --ncomponents 1



#
#
#

T=60
k=1000
l=2
for phone1 in `seq 0 46` ; do
   lower=$(( 1 + $phone1 ))
  for phone2 in `seq $lower 47` ; do
    echo $phone1 $phone2
   
    python $local/train_pegasos.py --phones $datadir/phone.list\
    --data_prefix $datadir\
    --data_suffix train_examples.npy\
    --in_prefix $old_exp\
    --in_suffix 1C.npy\
    --phone1 $phone1 --phone2 $phone2 --T $T --k $k --l $l \
    --out_w $exp/pegasos_w_${phone1}_${phone2}_${T}T_${k}k_${l}l.npy

done
done

#
# collect together all of the models
rm -rf $exp/pegasos_w_list_${T}T_${k}k_${l}l
for phone1 in `seq 0 46` ; do
   lower=$(( 1 + $phone1 ))
  for phone2 in `seq $lower 47` ; do
    echo $phone1 $phone2 $exp/pegasos_w_${phone1}_${phone2}_${T}T_${k}k_${l}l.npy >> $exp/pegasos_w_list_${T}T_${k}k_${l}l

done
done


#
# now we test
#



python $local/test_pegasos.py --phones $datadir/phone.list\
    --data_prefix $datadir\
    --data_suffix dev_examples.npy\
    --model_list $exp/pegasos_w_list_${T}T_${k}k_${l}l \
    --out_confusion_matrix $exp/confusion_matrix_dev_1C_${T}T_${k}k_${l}l.npy\
    --out_known_pair_confusion_matrix $exp/known_pair_confusion_matrix_dev_1C_${T}T_${k}k_${l}l.npy\
    --out_leehon_confusion_matrix $exp/leehon_confusion_matrix_dev_1C_${T}T_${k}k_${l}l.npy\
    --leehon_mapping $conf/phones.48-39

# the output of the pegasos test is contained here
python $local/print_leehon_results.py $exp/leehon_confusion_matrix_dev_1C_${q}Q.npy > $exp/leehon_confusion_matrix_dev_output_1C_${q}Q.txt

python $local/plot_threshold_known_comparison_diffs.py --covariance_factor .25\
    --kp_confusion_no_threshold $old_exp/known_pair_confusion_matrix_dev_1C_0Q.npy\
    --compare_quantiles 1 \
    --compare_prefix $exp/known_pair_confusion_matrix_dev_\
    --compare_suffix C_${T}T_${k}k_${l}l.npy\
    --plot_prefix $exp/known_pair_confusion_matrix_compare_1C_0Q_simple_\
    --plot_suffix C_pegasos_${T}T_${k}k_${l}l.png\
    --ytop .075\
    --xradius 100

#
# we now perform some analysis of the weight vectors
# to do so we make use of some scripts that are contained
# within the scripting directory
pegasos_local=`pwd`/scripts/pegasos_local
mkdir -p $pegasos_local

# in order to perform the comparisons we first collect a
# list containing paths to the weight vectors learned
# by  Pegasos and the order that they are in indicates
# the pairing with respect to phones



for phone1 in `seq 0 46` ; do
   lower=$(( 1 + $phone1 ))
  for phone2 in `seq $lower 47` ; do
done
done


# multiclass warm-start
cat << "EOF" > $conf/pegasos.conf
[EM]
n_init=10
n_iter=300
random_seed=0
tol=1e-6
min_data_count=30
EOF

for ncomponents in 2 3 6 9 12 ; do
python $local/estimate_all_models.py --phones $datadir/phone.list\
    --config $conf/pegasos.conf\
    --data_prefix ${datadir}\
    --data_suffix train_examples.npy\
    --out_prefix $exp\
    --out_suffix ${ncomponents}C.npy\
    --ncomponents ${ncomponents}\
    -v
done


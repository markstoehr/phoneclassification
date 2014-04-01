datadir=`pwd`/data/local/data
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf
old_exp=`pwd`/exp/pairwise_bernoulli_thresh
exp=`pwd`/exp/pegasos
scripts=`pwd`/scripts
pegasos_local=$scripts/pegasos_local
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

# now running pegasos
rm -f $exp/train_2C_warm_pegasos_avgs_meta_fl
for lambda_power in `seq 4 -1 -5` ; do
   lambda=`echo "" | awk "END { print (3.0)^($lambda_power) }"`
   for do_projection in True False ; do
       for regularize_diffs in True False ; do
           echo 2C_${lambda}lambda_${do_projection}Prj_${regularize_diffs}RD avgs_2C.npy meta_2C.npy $lambda $do_projection $regularize_diffs >> $exp/train_2C_warm_pegasos_avgs_meta_fl
       done
    done
done

python $pegasos_local/multicomponent_simple_pegasos_train.py --rootdir /home/mark/Research/phoneclassification \
    --confdir conf\
    --datadir data/local/data\
    --expdir exp/pegasos\
    --leehon_phones phones.48-39\
    --train_data_suffix train_examples.npy\
    --nrounds_multiplier 2\
    --avgs_meta_list_fl train_2C_warm_pegasos_avgs_meta_fl\
    --save_prefix warm_train_pegasos\
    --save_suffix .npy


rm -f $exp/train_21C_warm_pegasos_avgs_meta_fl
for lambda_power in `seq 3 -1 -5` ; do
   lambda=`echo "" | awk "END { print (3.0)^($lambda_power) }"`
   for do_projection in True False ; do
       for regularize_diffs in False ; do
           echo 21C_${lambda}lambda_${do_projection}Prj_${regularize_diffs}RD avgs_21C.npy meta_21C.npy $lambda $do_projection $regularize_diffs >> $exp/train_21C_warm_pegasos_avgs_meta_fl
       done
    done
done

python $pegasos_local/multicomponent_simple_pegasos_train.py --rootdir /home/mark/Research/phoneclassification \
    --confdir conf\
    --datadir data/local/data\
    --expdir exp/pegasos\
    --leehon_phones phones.48-39\
    --train_data_suffix train_examples.npy\
    --nrounds_multiplier 4\
    --avgs_meta_list_fl train_21C_warm_pegasos_avgs_meta_fl\
    --save_prefix warm_train_pegasos\
    --save_suffix .npy


python $local/simple_train_multicomponent_pegasos.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --model_avgs $exp/avgs_6C.npy \
  --model_meta $exp/meta_6C.npy \
  --save_prefix $exp/multicomponent_pegasos_6C \
  -l .05 .01 .1 .5 .005 .0001 \
  --eta .1 \
  -T 100000 100000 100000 100000 100000


# now we test
python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.05l_100000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.05l_100000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.05l_200000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.05l_200000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.05l_300000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.05l_300000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.05l_400000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.05l_400000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.05l_500000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.05l_500000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.05l_600000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.05l_600000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.05l_700000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.05l_700000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.05l_800000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.05l_800000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.01l_100000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.01l_100000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.01l_200000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.01l_200000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.01l_300000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.01l_300000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.01l_400000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.01l_400000T

python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.01l_500000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.01l_500000T

for i in 1 2 3 4 5 ; do
echo $i
python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_9C_0.05l_${i}00000T_W.npy \
  --W_meta $exp/meta_9C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_9C_0.05l_${i}00000T
done

for i in 1 2 3 4 5 ; do
echo $i
python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.1l_${i}00000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.1l_${i}00000T
done

for i in 11 12 13 14 15 ; do
echo $i
python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.05l_${i}00000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.05l_${i}00000T
done


for i in 6 7 8 9 10 ; do
echo $i
python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.01l_${i}00000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.01l_${i}00000T
done

for i in 6 7 8 9 ; do
echo $i
python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_9C_0.05l_${i}00000T_W.npy \
  --W_meta $exp/meta_9C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_9C_0.05l_${i}00000T
done

for i in 1 2 3 4 5 6 7  ; do
echo $i
python $local/test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_9C_0.01l_${i}00000T_W.npy \
  --W_meta $exp/meta_9C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_9C_0.01l_${i}00000T
done


python $local/fast_train_multicomponent_pegasos.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --model_avgs $exp/avgs_6C.npy \
  --model_meta $exp/meta_6C.npy \
  --save_prefix $exp/multicomponent_pegasos_6C \
  -l .05 \
  --niter 8

python $local/further_updates_train_multicomponent_pegasos.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.05l_500000T_W.npy \
  --model_meta $exp/meta_6C.npy \
  --save_prefix $exp/multicomponent_pegasos_6C \
  --start_t 500001 \
  -l .05 \
  --eta .1 \
  -T 100000 100000 100000 100000 100000


python $local/simple_train_multicomponent_pegasos.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --model_avgs $exp/avgs_6C.npy \
  --model_meta $exp/meta_6C.npy \
  --save_prefix $exp/multicomponent_pegasos_6C \
  -l .1 \
  --eta .1 \
  -T 100000 100000 100000 100000 100000


python $local/further_updates_train_multicomponent_pegasos.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.1l_500000T_W.npy \
  --model_meta $exp/meta_6C.npy \
  --save_prefix $exp/multicomponent_pegasos_6C \
  --start_t 500001 \
  -l .1 \
  --eta .1 \
  -T 100000 100000 100000 100000 100000

python $local/further_updates_train_multicomponent_pegasos.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.01l_500000T_W.npy \
  --model_meta $exp/meta_6C.npy \
  --save_prefix $exp/multicomponent_pegasos_6C \
  --start_t 500001 \
  -l .01 \
  --eta .1 \
  -T 100000 100000 100000 100000 100000


python $local/simple_train_multicomponent_pegasos.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --model_avgs $exp/avgs_9C.npy \
  --model_meta $exp/meta_9C.npy \
  --save_prefix $exp/multicomponent_pegasos_9C \
  -l .05 .01 .1 .005 .0001 \
  --eta .1 \
  -T 100000 100000 100000 100000 100000 100000 100000 100000 100000

# test the model

python $local/full_test_pegasos_svm_simple.py --root_dir /home/mark/Research/phoneclassification \
  --data_dir data/local/data \
  --W $exp/multicomponent_pegasos_6C_0.05l_1500000T_W.npy \
  --W_meta $exp/meta_6C.npy \
  --out_results_prefix $exp/multicomponent_pegasos_6C_0.06l_1500000T

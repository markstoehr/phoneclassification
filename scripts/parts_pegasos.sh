datadir=`pwd`/data/local/data
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf
old_exp=`pwd`/exp/pairwise_bernoulli_thresh
exp=`pwd`/exp/parts_pegasos
scripts=`pwd`/scripts
pegasos_local=$scripts/pegasos_local
mkdir -p $exp
mkdir -p $conf
plots=$exp/plots
mkdir -p $plots

train_examples=""
train_examples_S=""
save_parts=""
spec_save_parts=""
viz_spec_parts=""
phns=`cat $datadir/phone.list`

for phn in `cat $datadir/phone.list` ; do
  dev_examples="$dev_examples data/local/data/${phn}_dev_examples.npy"
  train_examples="$train_examples data/local/data/${phn}_train_examples.npy"
  train_examples_S="$train_examples_S data/local/data/${phn}_train_examples_S.npy"
  save_parts="$save_parts $exp/${phn}_train_parts.npy"
  spec_save_parts="$spec_save_parts $exp/${phn}_train_parts_S.npy"
  viz_spec_parts="$viz_spec_parts $exp/${phn}_train_parts_S.png"

done


# extract the parts
python $local/CExtractPatches.py --config $conf/main.config \
   --data $train_examples \
   --data_spec $train_examples_S \
   --save_parts $exp/patches_50_2r_100000max.npy \
   --spec_save_parts $exp/patches_50_2r_100000max_spec.npy \
   --viz_spec_parts $exp/patches_50_2r_100000max_viz.png \
   --n_components 50 \
   --patch_radius 2  -v


# code the utterance with the parts
dev_examples=""
templates=""
for phn in `cat ${datadir}/phone.list` ; do
   echo phn=${phn}
   python local/CGetParts.py --data $datadir/${phn}_train_examples.npy\
      --parts $exp/patches_50_2r_20000max.npy\
      --out ${exp}/${phn}_train_examples_parts.npy

   python local/CGetParts.py --data $datadir/${phn}_dev_examples.npy\
      --parts $exp/patches_50_2r_20000max.npy\
      --out ${exp}/${phn}_dev_examples_parts.npy



   dev_examples="$dev_examples ${exp}/${phn}_dev_examples_parts.npy"

done

python $local/Construct_sparse_dataset.py --leehon $conf/phones.48-39\
    --data $exp/ \
    --data_suffix train_examples_parts.npy \
    --out_prefix $exp/ \
    --out_suffix bsparse.npy

python $local/Construct_sparse_dataset.py --leehon $conf/phones.48-39\
    --data $exp/ \
    --data_suffix dev_examples_parts.npy \
    --out_prefix $exp/ \
    --out_suffix dev_bsparse.npy


# multiclass warm-start
cat << "EOF" > $conf/pegasos.conf
[EM]
n_init=15
n_iter=300
random_seed=0
tol=1e-6
min_data_count=30
EOF


for ncomponents in 2 3 6 9 12 15 ; do
python $local/estimate_all_models.py --phones $datadir/phone.list\
    --config $conf/pegasos.conf\
    --data_prefix ${exp}\
    --data_suffix train_examples_parts.npy\
    --out_prefix $exp\
    --out_suffix ${ncomponents}C.npy\
    --ncomponents ${ncomponents}\
    -v
done


   templates="$templates ${exp}/${phn}_parts_templates.npy"


   python local/CBernoulliMMTrain.py --config conf/main.config \
      --input_data_matrix $exp/${phn}_train_examples_parts.npy \
      --templates ${exp}/${phn}_parts_templates.npy \
      --weights ${exp}/${phn}_parts_weights.npy \
      --underlying_data ${data}/${phn}_train_examples_S.npy \
      --spec_templates ${exp}/${phn}_parts_S_templates.npy \
      --viz_spec_templates ${exp_dir}/${phn}_parts_S_templates -v




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


#
# now do the fast components
#

python $local/fast_multicomponent48_pegasos_training.py --root_dir /home/mark/Research/phoneclassification \
	--data_dir $exp/ \
        --use_sparse_suffix bsparse.npy \
        --dev_sparse_suffix dev_bsparse.npy \
	--model_avgs $exp/avgs_2C.npy \
	--model_meta $exp/meta_2C.npy \
	--save_prefix $exp/W_fast_pegasos_parts_all_useproj_1tsc_9C \
	-l .05 \
        --niter 10 --time_scaling 1.0 \
        --use_hinge 1 

python $local/fast_multicomponent48_pegasos_training.py --root_dir /home/mark/Research/phoneclassification \
	--data_dir $exp/ \
        --use_sparse_suffix bsparse.npy \
        --dev_sparse_suffix dev_bsparse.npy \
	--model_W $exp/W_fast_pegasos_parts_all_useproj_1tsc_9C_0.05l_8niter_W.npy \
	--model_meta $exp/meta_2C.npy \
	--save_prefix $exp/W_fast_pegasos_parts_all_useproj_1tsc_2C \
	-l .02 .01 .001 \
        --niter 5 --time_scaling 1.0 \
        --use_hinge 1 

python $local/fast_multicomponent48_pegasos_training.py --root_dir /home/mark/Research/phoneclassification \
	--data_dir $exp/ \
        --use_sparse_suffix bsparse.npy \
        --dev_sparse_suffix dev_bsparse.npy \
	--model_W $exp/W_fast_pegasos_parts_all_useproj_1tsc_9C_0.05l_8niter_W.npy \
	--model_meta $exp/meta_2C.npy \
	--save_prefix $exp/W_fast_pegasos_parts_all_useproj_1tsc_2C \
	-l .07 .09 .1 \
        --niter 10 --time_scaling 1.0 \
        --use_hinge 1 


python $local/fast_multicomponent48_pegasos_training.py --root_dir /home/mark/Research/phoneclassification \
	--data_dir $exp/ \
        --use_sparse_suffix bsparse.npy \
        --dev_sparse_suffix dev_bsparse.npy \
	--model_W $exp/W_fast_pegasos_parts_all_useproj_1tsc_2C_0.09l_6niter_W.npy \
	--model_meta $exp/meta_2C.npy \
	--save_prefix $exp/W_fast_pegasos_parts_all_useproj_1tsc_2C \
	-l .12 .14 .16 \
        --do_projection \
        --start_t 300000 \
        --niter 10 --time_scaling 1.0 \
        --use_hinge 1 

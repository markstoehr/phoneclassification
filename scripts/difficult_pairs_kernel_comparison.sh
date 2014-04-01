dir=`pwd`
exp=$dir/exp/difficult_pairs_kernel_comparison
mkdir -p $exp
local=$dir/local
data=$dir/data

python $local/load_compare_phone_sets_msc_kernel.py --phn_set1 ah ax \
    --phn_set2 ih ix --data_path $data --save_prefix $exp/ah_axVih_ix_kernel_svm_msc_compare

python $local/load_compare_phone_sets_msc_kernel.py --phn_set1 z \
    --phn_set2 s --data_path $data --save_prefix $exp/zVs_kernel_svm_msc_compare

python $local/load_compare_phone_sets_msc_kernel.py --phn_set1 m \
    --phn_set2 en n --data_path $data --save_prefix $exp/mVen_n_kernel_svm_msc_compare

python $local/load_compare_phone_sets_msc_kernel.py --phn_set1 er \
    --phn_set2 r --data_path $data --save_prefix $exp/erVr_kernel_svm_msc_compare

python $local/load_compare_phone_sets_msc_kernel.py --phn_set1 w \
    --phn_set2 el l --data_path $data --save_prefix $exp/wVel_l_kernel_svm_msc_compare

python $local/load_compare_phone_sets_msc_kernel.py --phn_set1 v \
    --phn_set2 cl epi sil vcl --data_path $data --save_prefix $exp/vVcl_epi_sil_vcl_kernel_svm_msc_compare


python $local/load_compare_phone_sets_msc_kernel.py --phn_set1 t \
    --phn_set2 k --data_path $data --save_prefix $exp/tVk_kernel_svm_msc_compare

python $local/load_compare_phone_sets_msc_kernel.py --phn_set1 eh \
    --phn_set2 ae --data_path $data --save_prefix $exp/ehVae_kernel_svm_msc_compare

python $local/load_compare_phone_sets_msc_kernel.py --phn_set1 ey \
    --phn_set2 iy --data_path $data --save_prefix $exp/eyViy_kernel_svm_msc_compare

python $local/load_compare_phone_sets_msc_kernel.py --phn_set1 b \
    --phn_set2 p --data_path $data --save_prefix $exp/bVp_kernel_svm_msc_compare

python $local/load_compare_phone_sets_msc_kernel.py --phn_set1 d \
    --phn_set2 g --data_path $data --save_prefix $exp/dVg_kernel_svm_msc_compare

#
# now we run experiments with spectrograms
#
#

data=$dir/data/local/data

python $local/load_compare_phone_sets_spec_kernel.py --phn_set1 ah ax \
    --phn_set2 ih ix --data_path $data --save_prefix $exp/ah_axVih_ix_kernel_svm_spec_compare

python $local/load_compare_phone_sets_spec_kernel.py --phn_set1 z \
    --phn_set2 s --data_path $data --save_prefix $exp/zVs_kernel_svm_spec_compare

python $local/load_compare_phone_sets_spec_kernel.py --phn_set1 m \
    --phn_set2 en n --data_path $data --save_prefix $exp/mVen_n_kernel_svm_spec_compare

python $local/load_compare_phone_sets_spec_kernel.py --phn_set1 er \
    --phn_set2 r --data_path $data --save_prefix $exp/erVr_kernel_svm_spec_compare

python $local/load_compare_phone_sets_spec_kernel.py --phn_set1 w \
    --phn_set2 el l --data_path $data --save_prefix $exp/wVel_l_kernel_svm_spec_compare

python $local/load_compare_phone_sets_spec_kernel.py --phn_set1 v \
    --phn_set2 cl epi sil vcl --data_path $data --save_prefix $exp/vVcl_epi_sil_vcl_kernel_svm_spec_compare


python $local/load_compare_phone_sets_spec_kernel.py --phn_set1 t \
    --phn_set2 k --data_path $data --save_prefix $exp/tVk_kernel_svm_spec_compare

python $local/load_compare_phone_sets_spec_kernel.py --phn_set1 eh \
    --phn_set2 ae --data_path $data --save_prefix $exp/ehVae_kernel_svm_spec_compare

python $local/load_compare_phone_sets_spec_kernel.py --phn_set1 ey \
    --phn_set2 iy --data_path $data --save_prefix $exp/eyViy_kernel_svm_spec_compare

python $local/load_compare_phone_sets_spec_kernel.py --phn_set1 b \
    --phn_set2 p --data_path $data --save_prefix $exp/bVp_kernel_svm_spec_compare

python $local/load_compare_phone_sets_spec_kernel.py --phn_set1 d \
    --phn_set2 g --data_path $data --save_prefix $exp/dVg_kernel_svm_spec_compare

#!/bin/bash

#
# First part of this experiment
# worked on extracted the feautres using a matlab script
# from the provided matlab code

local/get_scat_features.m

#
# then this will extract features to construct data sets that
# can then be fed into liblinear

data=`pwd`/data
exp=`pwd`/exp/scat_msc_exp
local=`pwd`/local
mkdir -p $exp

python $local/convert_kernel_mat_to_data.py --out_fl $exp/train.txt\
   --data_files $data/phone_fl_mfcc_features_list_0.txt

python $local/convert_kernel_mat_to_data.py --out_fl $exp/test.txt\
   --data_files $data/phone_fl_mfcc_features_list_1.txt

python $local/convert_kernel_mat_to_data.py --out_fl $exp/dev.txt\
   --data_files $data/phone_fl_mfcc_features_list_2.txt

#
# use liblinear to test the data
#
#

export PATH=$PATH:/var/tmp/stoehr/Software/liblinear-1.94
for C_exp in `seq -7 -1 -20` ; do
    C=`echo "2^$C_exp" | bc -l`
    echo $C $C_exp
    train -s 1 -e 0.0001 -c $C $exp/train.txt $exp/train_model_1_0.0001_${C_exp}    
    train -s 3 -e 0.0001 -c $C $exp/train.txt $exp/train_model_3_0.0001_${C_exp}    

done

for C_exp in -9.5 -9.75 ; do
    C=`echo "" | awk "END { print (2.0)^($C_exp) }"`
    echo $C $C_exp
    train -s 1 -e 0.0001 -c $C $exp/train.txt $exp/train_model_1_0.0001_${C_exp}    
    

done




for i in -9.5 -9.75 ; do 
/var/tmp/stoehr/Software/liblinear-1.94/predict $exp/dev.txt $exp/train_model_1_0.0001_${i}  $exp/dev_out_1_0.0001_${i}
done

ls $exp/*model* | sed 's:train_model: :' | awk '{ print $2 }' > $exp/model_params

for param_set in `cat $exp/model_params` ; do
    /var/tmp/stoehr/Software/liblinear-1.94/predict $exp/dev.txt $exp/train_model${param_set}  $exp/dev_out${param_set}
done

conf=`pwd`/conf
sort $conf/phones.48-39 | awk '{ print $1, NR }' > $exp/phns2ids

for param_set in `cat $exp/model_params` ; do
    python local/score_liblinear_predict_fl.py --phns2ids $exp/phns2ids\
   --leehon_mapping conf/phones.48-39\
   --predicted_labels $exp/dev_out${param_set}\
   --true_labels $exp/dev.txt >  $exp/dev_accuracy${param_set}
   echo ${param_set} `head $exp/dev_accuracy${param_set}`
done

#
#
#  Running the same experiment but over HOG features
#
#
    
exp=${exp}_hog
conf=`pwd`/conf
mkdir -p $exp


sort $conf/phones.48-39 | awk '{ print $1, NR }' > $exp/phns2ids

  

set_id=0
for data_set in train test dev ; do
  echo $data_set
  # python $local/compute_hog_features_for_list.py --infile $data/phone_fl_mfcc_features_25t_2s_40f_list_${set_id}.txt\
  #   --output_prefix $exp/HOG_${data_set}_data --phns2ids $exp/phns2ids\
  #   --output_fl_list $exp/HOG_features_${data_set}.txt\
  #   --ftype ascii --feature_stride 40 --nnuisance_dimensions 1

  python $local/convert_kernel_mat_to_data.py --out_fl $exp/${data_set}.txt\
   --data_files $exp/HOG_features_${data_set}.txt --phns2ids $exp/phns2ids\
   --ftype npy 

  set_id=$(( $set_id + 1))
done




export PATH=$PATH:/var/tmp/stoehr/Software/liblinear-1.94
for C_exp in `seq -3 1 0 ` ; do
    C=`echo "5^$C_exp" | bc -l`
    echo $C $C_exp
    train -s 1 -e 0.0001 -c $C $exp/train.txt $exp/train_model_1_0.0001_${C_exp}    
    train -s 3 -e 0.0001 -c $C $exp/train.txt $exp/train_model_3_0.0001_${C_exp}    

done



# convert output files into correct format for scoring
C_exp=-3
C=`echo "5^$C_exp" | bc -l`
/var/tmp/stoehr/Software/liblinear-1.94/predict $exp/dev.txt $exp/train_model_1_0.0001_${C_exp}  $exp/dev_out_1_0.0001_${C_exp}

/var/tmp/stoehr/Software/liblinear-1.94/predict $exp/dev.txt $exp/train_model_3_0.0001_${C_exp}  $exp/dev_out_3_0.0001_${C_exp}



python local/score_liblinear_predict_fl.py --phns2ids $exp/phns2ids\
   --leehon_mapping conf/phones.48-39\
   --predicted_labels $exp/dev_out_1_0.0001_${C_exp}\
   --true_labels $exp/dev.txt

python local/score_liblinear_predict_fl.py --phns2ids $exp/phns2ids\
   --leehon_mapping conf/phones.48-39\
   --predicted_labels $exp/dev_out_3_0.0001_${C_exp}\
   --true_labels $exp/dev.txt

C_exp=-2
/var/tmp/stoehr/Software/liblinear-1.94/predict $exp/dev.txt $exp/train_model_1_0.0001_${C_exp}  $exp/dev_out_1_0.0001_${C_exp}

python local/score_liblinear_predict_fl.py --phns2ids $exp/phns2ids\
   --leehon_mapping conf/phones.48-39\
   --predicted_labels $exp/dev_out_1_0.0001_${C_exp}\
   --true_labels $exp/dev.txt

for C_exp in `seq -3 1 0 ` ; do
    C=`echo "5^$C_exp" | bc -l`
    echo $C $C_exp
    /var/tmp/stoehr/Software/liblinear-1.94/predict $exp/dev.txt $exp/train_model_1_0.0001_${C_exp}  $exp/dev_out_1_0.0001_${C_exp}

   python local/score_liblinear_predict_fl.py --phns2ids $exp/phns2ids\
   --leehon_mapping conf/phones.48-39\
   --predicted_labels $exp/dev_out_1_0.0001_${C_exp}\
   --true_labels $exp/dev.txt

   /var/tmp/stoehr/Software/liblinear-1.94/predict $exp/dev.txt $exp/train_model_3_0.0001_${C_exp}  $exp/dev_out_3_0.0001_${C_exp}

   python local/score_liblinear_predict_fl.py --phns2ids $exp/phns2ids\
   --leehon_mapping conf/phones.48-39\
   --predicted_labels $exp/dev_out_3_0.0001_${C_exp}\
   --true_labels $exp/dev.txt

done


for C_exp in `seq -4 -1 -7 ` ; do
    C=`echo "5^$C_exp" | bc -l`
    echo $C $C_exp
    train -s 2 -e 0.0001 -c $C $exp/train.txt $exp/train_model_2_0.0001_${C_exp}    
    train -s 1 -e 0.0001 -c $C $exp/train.txt $exp/train_model_1_0.0001_${C_exp}    

done

for C_exp in `seq -4 -1 -7 ` ; do
    C=`echo "5^$C_exp" | bc -l`
    echo $C $C_exp
    echo Testing -s 1
    /var/tmp/stoehr/Software/liblinear-1.94/predict $exp/dev.txt $exp/train_model_1_0.0001_${C_exp}  $exp/dev_out_1_0.0001_${C_exp}

   python local/score_liblinear_predict_fl.py --phns2ids $exp/phns2ids\
   --leehon_mapping conf/phones.48-39\
   --predicted_labels $exp/dev_out_1_0.0001_${C_exp}\
   --true_labels $exp/dev.txt

    echo Testing -s 2
   /var/tmp/stoehr/Software/liblinear-1.94/predict $exp/dev.txt $exp/train_model_2_0.0001_${C_exp}  $exp/dev_out_2_0.0001_${C_exp}


   python local/score_liblinear_predict_fl.py --phns2ids $exp/phns2ids\
   --leehon_mapping conf/phones.48-39\
   --predicted_labels $exp/dev_out_2_0.0001_${C_exp}\
   --true_labels $exp/dev.txt

done

####
#
#
# kernel matrix work
#
#
#

exp=`pwd`/exp/scat_msc_kernel
data=`pwd`/data
conf=`pwd`/conf
local=`pwd`/local


python $local/do_all_liblinear_matlab.py --feat_dim 139868\
    --save_prefix $exp/kr_local_scalingNN7\
    --penalty_list .00001 .001 .01 .1 1\
    --kr_list $data/kernel_matrix_fls_classes_localscaleNN7_0.txt

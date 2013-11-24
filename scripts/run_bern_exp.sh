#!/bin/bash -ex

data=data/local/data
exp_dir=exp/bernoulli_all_phones
phn_ids_fl=$exp_dir/phn_ids
dev_examples=""
train_examples=""
templates=""
phn_id=0
rm -f $phn_ids_fl
touch $phn_ids_fl
for phn in `cat ${data}/phone.list` ; do
  dev_examples="$dev_examples ${data}/${phn}_dev_examples.npy"
  train_examples="$train_examples ${data}/${phn}_train_examples.npy"
  templates="$templates ${exp_dir}/${phn}_templates.npy"
  echo $phn_id $phn >> $phn_ids_fl
  phn_id=$(( $phn_id + 1))
done

python local/CTestBernoulli.py --data $dev_examples --templates $templates\
    --out_scores ${exp_dir}/bernoulli_dev_scores.npy\
    --out_labels ${exp_dir}/bernoulli_dev_labels.npy\
    --out_components ${exp_dir}/bernoulli_dev_components.npy\
    --out_top_predicted_labels ${exp_dir}/bernoulli_dev_top_predicted_labels.npy\
    --out_top_predicted_components ${exp_dir}/bernoulli_dev_top_predicted_components.npy\
    --out_confusion_matrix $exp_dir/bernoulli_dev_confusion.npy\
    --out_leehon_confusion_matrix $exp_dir/bernoulli_dev_leehon_confusion.npy\
    --leehon_mapping conf/phones.48-39\
    --phn_ids $phn_ids_fl


python local/CTestBernoulli.py --data $train_examples --templates $templates\
    --out_scores ${exp_dir}/bernoulli_train_scores.npy\
    --out_labels ${exp_dir}/bernoulli_train_labels.npy\
    --out_components ${exp_dir}/bernoulli_train_components.npy\
    --out_top_predicted_labels ${exp_dir}/bernoulli_train_top_predicted_labels.npy\
    --out_top_predicted_components ${exp_dir}/bernoulli_train_top_predicted_components.npy\
    --out_confusion_matrix $exp_dir/bernoulli_train_confusion.npy\
    --out_leehon_confusion_matrix $exp_dir/bernoulli_train_leehon_confusion.npy\
    --leehon_mapping conf/phones.48-39\
    --phn_ids $phn_ids_fl




for dset in dev train ; do
python local/confusion_matrix_to_error_rate.py --cmat $exp_dir/bernoulli_${dset}_leehon_confusion.npy --strformat 1.4 > $exp_dir/bernoulli_${dset}_leehon_errorrate.txt

python local/confusion_count_histogram.py --cmat $exp_dir/bernoulli_${dset}_leehon_confusion.npy --add_const .1 --nbins 20\
   --title_full "BERNOULLI  Confusions"\
   --title_mistakes "BERNOULLI Mistaken Confusions"\
   --out_full $exp_dir/bernoulli_${dset}_leehon_confusion_histogram.png\
   --out_mistakes $exp_dir/bernoulli_${dset}_leehon_confusion_mistakes_histogram.png

python local/confusion_threshold_visualization.py --cmat $exp_dir/bernoulli_${dset}_leehon_confusion.npy --npoints 30\
   --title_full "BERNOULLI  Confusions"\
   --title_mistakes "BERNOULLI  Mistaken Confusions"\
   --out_full $exp_dir/bernoulli_${dset}_leehon_confusion_thresh_viz.png\
   --out_mistakes $exp_dir/bernoulli_${dset}_leehon_confusion_mistakes_thresh_viz.png

python local/make_rcm.py --cmat $exp_dir/bernoulli_${dset}_leehon_confusion.npy \
   --phns data/local/data/phone.list \
   --leehon_convert conf/phones.48-39   --verbose \
   --just_show_mistakes    --threshold 40.0 \
   --title "BERNOULLI Mistake Confusions" \
   --save_path $exp_dir/bernoulli_${dset}_leehon_confusion_rcm_mistakes.png


python local/make_rcm.py --cmat $exp_dir/bernoulli_${dset}_leehon_confusion.npy \
   --phns data/local/data/phone.list \
   --leehon_convert conf/phones.48-39   --verbose \
   --threshold 40.0 \
   --title "BERNOULLI Confusions" \
   --save_path $exp_dir/bernoulli_${dset}_leehon_confusion_rcm.png --ceil 200

done

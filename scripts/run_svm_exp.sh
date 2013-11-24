#!/bin/bash -ex

datadir=data/local/data
exp_dir=exp/all_phones_exp
penalties="little_reg reg_plus reg_plus_plus reg_plus_plus_plus"

dev_examples=""
dev_lengths=""
phn_ids=""
phn_id=0
for phn in `cat $datadir/phone.list` ; do
  dev_examples="$dev_examples data/local/data/${phn}_dev_examples.npy"
  dev_lengths="$dev_lengths data/local/data/${phn}_dev.lengths"
  phn_ids="$phn_ids $phn_id"
  phn_id=$(( $phn_id + 1 ))
done


for penalty in $penalties ; do
    python local/CRunSVM.py \
	--coefs $exp_dir/svm_train_all_pairs_linear_${penalty}_coefs.npy\
  --intercepts $exp_dir/svm_train_all_pairs_linear_${penalty}_intercepts.npy\
  --out_ids $exp_dir/svm_train_all_pairs_${penalty}.ids\
  --input_data_matrices $dev_examples\
  --input_lengths $dev_lengths\
  --out_confusion_matrix $exp_dir/svm_all_pairs_${penalty}_confusion.npy\
  --out_leehon_confusion_matrix $exp_dir/svm_all_pairs_${penalty}_leehon_confusion.npy\
  --input_ids $phn_ids\
  --leehon_mapping conf/phones.48-39\
  --phn_ids $exp_dir/svm_train_all_pairs_${penalty}_phns.ids

   python local/confusion_matrix_to_error_rate.py --cmat $exp_dir/svm_all_pairs_${penalty}_leehon_confusion.npy --strformat 1.4 > $exp_dir/svm_${penalty}_leehon_errorrate.txt


python local/confusion_count_histogram.py --cmat $exp_dir/svm_all_pairs_${penalty}_leehon_confusion.npy --add_const .1 --nbins 20\
   --title_full "SVM All Pairs ${penalty} Confusions"\
   --title_mistakes "SVM All Pairs ${penalty} Mistaken Confusions"\
   --out_full $exp_dir/svm_all_pairs_${penalty}_leehon_confusion_histogram.png\
   --out_mistakes $exp_dir/svm_all_pairs_${penalty}_leehon_confusion_mistakes_histogram.png

python local/confusion_threshold_visualization.py --cmat $exp_dir/svm_all_pairs_${penalty}_leehon_confusion.npy --npoints 30\
   --title_full "SVM All Pairs ${penalty} Confusions"\
   --title_mistakes "SVM All Pairs ${penalty} Mistaken Confusions"\
   --out_full $exp_dir/svm_all_pairs_${penalty}_leehon_confusion_thresh_viz.png\
   --out_mistakes $exp_dir/svm_all_pairs_${penalty}_leehon_confusion_mistakes_thresh_viz.png

python local/make_rcm.py --cmat $exp_dir/svm_all_pairs_${penalty}_leehon_confusion.npy \
   --phns data/local/data/phone.list \
   --leehon_convert conf/phones.48-39   --verbose \
   --just_show_mistakes    --threshold 40.0 \
   --title "SVM All Pairs ${penalty} Mistake Confusions" \
   --save_path $exp_dir/svm_all_pairs_${penalty}_leehon_confusion_rcm_mistakes.png


python local/make_rcm.py --cmat $exp_dir/svm_all_pairs_${penalty}_leehon_confusion.npy \
   --phns data/local/data/phone.list \
   --leehon_convert conf/phones.48-39   --verbose \
   --threshold 40.0 \
   --title "SVM All Pairs ${penalty} Confusions" \
   --save_path $exp_dir/svm_all_pairs_${penalty}_leehon_confusion_rcm.png --ceil 200



done





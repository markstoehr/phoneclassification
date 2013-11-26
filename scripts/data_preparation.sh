TIMIT=/var/tmp/stoehr/timit
dir=`pwd`/data/local/data
mkdir -p $dir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

export KALDI_ROOT=/var/tmp/stoehr/Software/kaldi-trunk
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet-cpubin/:$KALDI_ROOT/src/kwsbin:$PWD:$PATH
export LC_ALL=C

export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe

rm -r links/ 2>/dev/null
mkdir links/



ln -s $TIMIT links

TrainDir=$TIMIT/train

out_dir=$dir/wav_files
mkdir -p $out_dir

dsets="train dev core_test"

out_train_dir=$out_dir/$dset
mkdir -p $out_train_dir
out_dev_dir=$out_dir/$dset
mkdir -p $out_dev_dir
out_core_test_dir=$out_dir/$dset
mkdir -p $out_core_test_dir



find -L $TrainDir \( -iname 's[ix]*.WAV' -o -iname 's[ix]*.wav' \) > $dir/train.flst
python $local/flist_to_scp.py < $dir/train.flst | sort > $dir/train_wav.scp
cat $dir/train_wav.scp | awk '{print $1}' > $dir/train.uttids
cat $dir/train_wav.scp | sed 's/wav/phn/'> $dir/train_phn.scp

echo '#!/bin/bash' > $dir/convert_train_wav.scp
sed "s:^:"$out_train_dir"/:" < $dir/train_wav.scp | sed "s: :.wav :" |awk '{printf("'$sph2pipe' -f wav %s %s\n", $2,$1);}' >> $dir/convert_train_wav.scp

awk '{print $NF}' $dir/train_wav.scp > $dir/train.wav
awk '{print $2}' $dir/train_phn.scp > $dir/train.phn



#
# setup the speakers for the dev and test sets
#
cat << "EOF" | sort > $conf/dev_spk
faks0
mmdb1
mbdg0
fedw0
mtdt0
fsem0
mdvc0
mrjm4
mjsw0
mteb0
fdac1
mmdm2
mbwm0
mgjf0
mthc0
mbns0
mers0
fcal1
mreb0
mjfc0
fjem0
mpdf0
mcsh0
mglb0
mwjg0
mmjr0
fmah0
mmwh0
fgjd0
mrjr0
mgwt0
fcmh0
fadg0
mrtk0
fnmr0
mdls0
fdrw0
fjsj0
fjmg0
fmml0
mjar0
fkms0
fdms0
mtaa0
frew0
mdlf0
mrcs0
majc0
mroa0
mrws1
EOF


cat << "EOF" | sort > $conf/core_test_spk
mdab0
mwbt0
felc0
mtas1
mwew0
fpas0
mjmp0
mlnt0
fpkt0
mlll0
mtls0
fjlm0
mbpm0
mklt0
fnlp0
mcmj0
mjdh0
fmgd0
mgrt0
mnjm0
fdhc0
mjln0
mpam0
fmld0
EOF


TestDir=$TIMIT/test
find -L $TestDir \( -iname 's[ix]*.WAV' -o -iname 's[ix]*.wav' \) > test.flst
$local/flist_to_scp.py < test.flst | sort > test_wav.scp
cat test_wav.scp | awk '{print $1}' > test.uttids
cat test_wav.scp | sed 's/wav/phn/'> test_phn.scp
awk '{print $2}' test_wav.scp > test.wav
awk '{print $2}' test_phn.scp > test.phn

$local/extract_by_spk.py -i test_wav.scp --spk $conf/dev_spk > dev_wav.scp
$local/extract_by_spk.py -i test_wav.scp --spk $conf/core_test_spk > core_test_wav.scp
$local/extract_by_spk.py -i test_phn.scp --spk $conf/dev_spk > dev_phn.scp
$local/extract_by_spk.py -i test_phn.scp --spk $conf/core_test_spk > core_test_phn.scp

awk '{print $1}' dev_wav.scp > dev.uttids
awk '{print $1}' core_test_wav.scp > core_test.uttids

awk '{print $2}' dev_wav.scp > dev.wav
awk '{print $2}' dev_phn.scp > dev.phn
awk '{print $2}' core_test_wav.scp > core_test.wav
awk '{print $2}' core_test_phn.scp > core_test.phn

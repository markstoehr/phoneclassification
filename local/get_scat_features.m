addpath('/var/tmp/stoehr/Software/scatnet')
addpath_scatnet

N = 2^13;
T_s = 2560;

filt1_opt.filter_type = {'gabor_1d','morlet_1d'};
filt1_opt.Q = [8 1];
filt1_opt.J = T_to_J(368,filt1_opt);

sc1_opt = struct();
sc1_opt.oversampling=2;

filters = filter_bank(N, filt1_opt);

x = wavread('/var/tmp/stoehr/timit_wav/train/dr1/fcjf0/sa1.wav');

filt1_opt.filter_type = {'gabor_1d','morlet_1d'};
filt1_opt.Q = [8 1];
filt1_opt.J = T_to_J(512,filt1_opt);


filters = filter_bank(N, filt1_opt);

sc1_opt.M = 2;

Wop = wavelet_factory_1d(N, filt1_opt, sc1_opt);
z = log_scat(renorm_scat(scat(x(3*N:4*N),Wop)));


z = wavelet_1d(x,filters{2},sc1_opt);

z = spec_freq_average(x,filters,sc1_opt);
scatt_fun = @(x)(format_scat(log_scat(spec_freq_average(x,filters,sc1_opt))));




src = phone_src('/var/tmp/stoehr/timit');

[train_set,test_set,valid_set] = phone_partition(src);

duration_fun = @(x,obj)(32*duration_feature(x,obj));

features = {scatt_fun,duration_fun};

for k = 1:length(features)
	fprintf('testing feature #%d...',k);
	tic;
	sz = size(features{k}(randn(N,1)));
	aa = toc;
	fprintf('OK (%.2fs) (size [%d,%d])\n',aa,sz(1),sz(2));
end

database_opt.input_sz = N;
database_opt.output_sz = T_s;
database_opt.obj_normalize = 2;
database_opt.collapse = 1;
database_opt.parallel = 0;
database_opt.file_normalize = [];

db = prepare_database(src,features,database_opt);
db.features = single(db.features);

save('/var/tmp/stoehr/phoneclassification/data/db.mat','db')
load('/var/tmp/stoehr/phoneclassification/data/db.mat')


set_id = 0;
for kernel_set = { train_set test_set valid_set }
  kernel_set = kernel_set{1}';
total_ind_used = 0
fl_name_fl = sprintf('/var/tmp/stoehr/phoneclassification/data/phone_fl_mfcc_features_list_%d.txt',set_id)
fid = fopen(fl_name_fl,'w');
for class_id = 1:length(db.src.classes)
    class_indices = find(([db.src.objects.class] == class_id) ...
    .* ([db.src.objects.subset] == set_id));
    total_ind_used = total_ind_used +  length(class_indices)
    class_name = db.src.classes{class_id};
    save_name = ...
 ...
...
sprintf('/var/tmp/stoehr/phoneclassification/data/msc_features_%s_%d.dat',class_name,set_id)
   fprintf(fid,'%s\n',save_name);
 X = double(db.features(:,class_indices))';
    save(save_name,'X','-ascii')
  end
  fclose(fid);
  disp(total_ind_used)
 set_id = set_id + 1
  end
    
  
save('/var/tmp/stoehr/phoneclassification/data/db_paper.mat','db')
kernel_set = 1:size(db.features,2);
kernel_type='gaussian';
kernel_format='square';

block_size = 4000;
norm1 = sum(abs(db.features.^2),1);
NN15 = zeros(15,size(db.features,2));
r=1
while r <= size(db.features,2)
  disp(r)
  ind = r:min(r+block_size-1,size(db.features,2));
  Kr = -2*db.features(:,train_set).'*db.features(:,ind);
  Kr = bsxfun(@plus,norm1(train_set).',Kr);
  Kr = bsxfun(@plus,norm1(ind),Kr);
  B = sort(Kr);
  NN15(:,ind)=B(1:15,:);
  r = min(r+block_size-1,size(db.features,2)) + 1;
end

save('/var/tmp/stoehr/phoneclassification/data/db_NN15.mat','NN15')

block_size = 3000;

NN7 = NN15(7,:).';


set_id=0
for kernel_set = { train_set test_set valid_set }
  
  fid = fopen(sprintf('/var/tmp/stoehr/phoneclassification/data/kernel_matrix_fls_classes_localscaleNN7_%d.txt',set_id),'w');
  
  kernel_set=kernel_set{1}';
  disp(size(kernel_set))

  vector_ct = length(kernel_set);
  for class_id=1:size(db.src.classes,2)
  
    ind = find(([db.src.objects.subset] == set_id) .* ([db.src.objects.class] == class_id));
    p = randperm(length(ind));
    ind = ind(p(1:min(block_size,length(p))));
    Kr = -2*db.features(:,ind).'*db.features(:,train_set);
    Kr = bsxfun(@plus,norm1(ind).',Kr);
    Kr = bsxfun(@plus,norm1(train_set),Kr);
    Kr = bsxfun(@times,NN7(train_set).',Kr);
    Kr = bsxfun(@times,NN7(ind),Kr);
    savename=sprintf('/var/tmp/stoehr/phoneclassification/data/Kr_local_scalNN7_%d_%s_%d_%d.mat',class_id,db.src.classes{class_id},set_id,block_size)
    fprintf(fid,'%d %s %d %s\n',class_id,db.src.classes{class_id},size(Kr,1),savename);
    save(savename,'Kr')

end
set_id = set_id + 1
fclose(fid);
end

% save the training labels
classes = [db.src.objects.class]';
set_id = 0
for kernel_set = { train_set test_set valid_set }
  fid = fopen(sprintf('/var/tmp/stoehr/phoneclassification/data/kernel_matrix_labels_fls_%d.txt',set_id),'w');
  kernel_set=kernel_set{1}';
  disp(size(kernel_set))

  vector_ct = length(kernel_set);


r=1;
while r < vector_ct
  ind = r:min(r+block_size-1,vector_ct);
  cur_class=classes([kernel_set(ind)]);
 ...
 ...
 savename=sprintf('/var/tmp/stoehr/phoneclassification/data/Kr_classes%d_%d_%d.dat',block_size,set_id,r)
 fprintf(fid,'%s\n',savename);
 save(savename,'cur_class','-ascii')
  r = r+length(ind)
end
set_id = set_id + 1
fclose(fid);
end



%
% extract the data
%
%

X = []

fid = fopen('data/local/data/train.wav');

tline = fgetl(fid);
while ischar(tline)
      [y,Fs] = wavread(tline);
      X = horzcat(X,scatt_fun(y));
      
      disp(tline)
      
      tline = fgetl(fid);
      
end
	
fclose(fid);


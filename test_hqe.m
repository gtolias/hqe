% This script reproduces the results of the paper "G. Tolias & H. Jegou. Visual query expansion
% with or without geometry - refining local descriptors by feature aggregation, 
% Pattern Recognition 2014". It creates the indexing structure and performs retrieval
% on Oxford5k for the variant called HQE in the paper. 

% download asmk package
if ~exist('asmk-master') 
  system('wget https://github.com/gtolias/asmk/archive/master.zip');
  system('unzip master.zip');
end
addpath('asmk-master');
% download yael
if ~exist('yael') 
  system('wget https://gforge.inria.fr/frs/download.php/file/34218/yael_matlab_linux64_v438.tar.gz');
  system('mkdir yael');
  system('tar -C yael/ -zxvf yael_matlab_linux64_v438.tar.gz');
end
addpath('yael');
% download required data
if ~exist('data') 
  system('wget -nH --cut-dirs=4 -r -Pdata/ ftp://ftp.irisa.fr/local/texmex/corpus/iccv2013/');
end
% compile mex files
if ~exist('scormex.mexa64')
  mex scormex.c scorc.c
end
if ~exist('aggregate_binary.mexa6')
  mex aggregate_binary.c aggregate.c CFLAGS='\$CFLAGS -fopenmp' LDFLAGS='\$LDFLAGS -fopenmp'
end

% config for dataset
cfg = config_oxford();

% parameters
prm.nbits 				= 128;        % dimension of binary signatures
prm.k 						= 2^16;       % codebook size
prm.ht 						= 48;         % hamming distance threshold, h_t
prm.ht_strict 		= 32;         % strict threshold, h_t^*
prm.hqealpha 		  = 0.5;        % parameter alpha
prm.top_n 			  = 100;        % short-list to apply HQE
prm.ma 				    = 3;          % descriptor soft-assignment to ma visual words
prm.cor_thres 		= 5;          % number of strong correspondences, c_t

docluster = false;  % compute codebook/used a pre-computed one
compute_vw = false; % compute visual words for test set/load pre-computed ones

% Load training descriptors
fprintf ('* Loading and post-processing training descriptors\n'); 
vtrain = load_ext(cfg.train_sift_fname, 128);
if ~docluster
  vwtrain = load_ext(cfg.train_vw_fname);
  codebook = load_ext(cfg.codebook_fname, 128);
end

% SIFT post processing, ROOT-SIFT and SHIFT-SIFT
[vtrain vtrain_mean] = desc_postprocess (vtrain);

% Learn the ivf structure
tic;
if docluster
  % Learn the IVF structure (and codebook)
  ivfhe = yael_ivf_he (prm.k, prm.nbits, single(vtrain), @yael_nn);
else
  % Learn the IVF structure
  % Learned codebook and visual words of training descriptors are provided
  ivfhe = yael_ivf_he (prm.k, prm.nbits, single(vtrain), @yael_nn, codebook, vwtrain);
end
fprintf ('* Learned the IVF structure in %.3f seconds\n', toc); 

% Load test descriptors and number of features per image
fprintf ('* Loading and post-processing database descriptors\n'); 
vtest = single (load_ext(cfg.test_sift_fname, 128));
nftest = load_ext(cfg.test_nf_fname);

% SIFT post processing, ROOT-SIFT and SHIFT-SIFT
vtest = desc_postprocess (vtest, vtrain_mean);

% Compute visual words for test descriptors
if compute_vw
  fprintf ('* Computing visual words for database descriptors\n'); 
  [vwtest, ~] = ivfhe.quantizer (ivfhe.quantizer_params, vtest);
else
  vwtest = load_ext(cfg.test_vw_fname);
end

% Compute image ids for all descriptors to be inserted in the ivf structure
cs = [1 cumsum(double (nftest)) + 1];
[~, ids] = histc (1: sum(nftest), cs); %image ids here

% Add descriptors to the ivf structure
tic;
[vwtest, codes] = ivfhe.add (ivfhe, int32(ids), vtest, vwtest);
vwperimage = accumarray(ids', vwtest', [max(ids),1], @(x){x});
fprintf ('* Added %d images to the IVF structure in %.3f seconds\n', numel(nftest), toc); 

% Weighting function for descriptor similarity
idx = [1:-2/ivfhe.nbits:-1];
scoremap = single (exp(- ((0:ivfhe.nbits)/(ivfhe.nbits/4)).^2));	% weighting function in the HQE paper

% Compute idf values
tic;
listw = single (compute_idf (vwtest, nftest, ivfhe.k));
listw = listw .^ 2; % squared to account idf for both images
fprintf ('* Computed idf values in %.3f seconds\n', toc); 

% Compute normalization factors for database images
tic;
normf = compute_norm_factor (vwtest, nftest, listw);
fprintf ('* Computed normalization factors in %.3f seconds\n', toc);

fprintf ('* Imbalance factor of inverted file %d\n', ivfhe.imbfactor());

% Save ivf
fivf_name = cfg.ivf_fname;
fprintf ('* Save the inverted file to %s\n', fivf_name);
ivfhe.save (ivfhe, fivf_name);

fprintf ('* Free the inverted file\n');
% Free the variables associated with the inverted file
yael_ivf ('free');
clear ivfhe;

% Save weighting function values, idf values and normalization factors for database images
save (sprintf ('%s_other.mat', fivf_name), 'scoremap', 'listw', 'normf', 'vwperimage');

% Clear training data
clear vtrain vwtrain;

	
% -----------------------------------
% Query inverted file
% -----------------------------------

% Load ivf
fprintf ('* Load the inverted file from %s\n', fivf_name);
ivfhe = yael_ivf_he (fivf_name);
load (sprintf ('%s_other.mat', fivf_name), 'scoremap', 'listw', 'normf', 'vwperimage');

ivfhe.scoremap = scoremap;
ivfhe.listw = listw;
ivfhe.normf = normf;
ivfhe.vwperimage = vwperimage;

% Load ground truth structure for Oxford5k
load (cfg.gnd_fname);

% Load test images and number of features per image, to be used a queries
vtest = single (load_ext(cfg.test_sift_fname, 128));
gtest = load_ext(cfg.test_geom_fname, 5);
nftest = load_ext(cfg.test_nf_fname);

% SIFT post processing, ROOT-SIFT and SHIFT-SIFT
vtest = desc_postprocess (vtest, vtrain_mean);

cs = [1 cumsum( double (nftest)) + 1];

fprintf ('* Perform queries\n');
% Query using 55 predefined bounding boxes on oxford images
for q=1:numel(qidx)
  
  fprintf ('* Loading and postprocessing query descriptors\n');	
  % Descriptors of q-th image
  dquery = vtest (:, cs(qidx(q)):cs(qidx(q)+1)-1);
  gquery = gtest (:, cs(qidx(q)):cs(qidx(q)+1)-1);
  cqidx = crop_query (gnd.bbx (q, :), gquery(1:2, :));
  dquery = dquery (:, cqidx);
 
  % Compute visual words for test descriptors
  tic;
  [vquery, ~] = ivfhe.quantizer (ivfhe.quantizer_params, dquery, prm.ma);
  fprintf ('* Computed visual words for query descriptors in %.3f seconds\n', toc);		
  
  vquery = reshape (vquery', [1 prm.ma * numel(cqidx)]);
  dquery = repmat (dquery, 1, prm.ma);
  nquery = size(dquery, 2);
	bquery = ivfhe.binsign (ivfhe, dquery, vquery);

  % Query ivf structure and collect matches
  tic;
  [matches] = ivfhe.query (ivfhe, int32(1:nquery), dquery, prm.ht, vquery, bquery);
  [matches, sim] = hqe(ivfhe, matches, ivfhe.scoremap(matches(3, :)+1), vquery, bquery, prm);
  fprintf ('* Performed query %d in %.3f seconds\n', q, toc);		

  % Compute final similarity score per image and rank
	[~, ranks(:, q)] = sort (scormex (uint32(matches(1,:))', uint32(matches (2,:))', single(sim), numel(ivfhe.normf), 1) ./ ivfhe.normf, 'descend');
end

% Compute mean Average Precision (mAP)
map = compute_map (ranks, gnd);
fprintf ('* mAP on Oxford5k is %.4f\n', map);

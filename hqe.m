function [matches, similarity] = hqe (ivfstruct, matches, similarity, qvw, qbs, prm) 

  % score = accumarray (matches(2,:)', similarity, [numel(ivfstruct.normf) 1]);
  score = scormex (uint32(matches(1,:))', uint32(matches (2,:))', single(similarity), numel(ivfstruct.normf), 1);
  passedcor = find (matches(3, :) <= prm.ht_strict);
  nc = accumarray (matches(2,passedcor)', 1, [numel(ivfstruct.normf) 1]); % number of strong correspondences
  [~, ridx] = sort (score, 'descend');

  % set of reliable images
  rel_ids = ridx (1:prm.top_n);
  rel_ids = rel_ids (find (nc (rel_ids) >= prm.cor_thres));
  if numel (rel_ids) == 0,  return; end

  % augmentation - select visual words from reliable images
  histj = zeros(1, ivfstruct.k); % joint histogram of binary occurrences
  for j = 1:numel(rel_ids)
    vw_reli{j} = ivfstruct.vwperimage{rel_ids(j)}';
    histj(vw_reli{j}) = histj(vw_reli{j}) + 1;
  end

  % selection of reliable visual words
  [sel_vw] = find_reliable_words(histj, ivfstruct.k, unique (qvw), prm.hqealpha);
  selvwh = accumarray(sel_vw', 1, [ivfstruct.k, 1])';

  % get needed binary signatures from ivfstruct
  for j = 1:numel(rel_ids)
    f = find (selvwh(vw_reli {j}));
    augvi{j} = [ vw_reli{j}(f); int32( ones(1, numel(f)) * rel_ids (j))]; 
  end
  augvimat = cell2mat(augvi);
  augvi1d = uint64(augvimat(1, :)) * 2^32 + uint64(augvimat(2, :)); % map to 1d for faster sorting
  
  % get binary vectors from the ivf
  [~, rs] = sort(augvi1d);
  augvw = augvimat(1, rs);
  augbs = ivfstruct.findbs(augvw, augvimat(2, rs)');

  % aggregate all bs (query+database ones) per visual word
  [sbin, qvwn] = aggregate_bs ([qvw augvw], [qbs augbs], ivfstruct.bin2compactbin, ivfstruct.k);
  
  % re-query ivf
  [matches, similarity] = ivfstruct.queryw (ivfstruct, int32 (1:numel (qvwn)), [], prm.ht, qvwn, uint8 (sbin));

function [sel_vw] = find_reliable_words(vw_hist_rel, k, quvw, alpha)

  % selected visual words
  fnz = find (vw_hist_rel); % find non zero
  nnz = numel (fnz); 
  [~, ri] = sort (vw_hist_rel(fnz), 'descend');

  histq = zeros(1, k);
  histq (quvw) = 1;
  newones = ~histq (fnz(ri)); % not present in the query
  fnew = find (newones);

  n = floor(alpha * numel(quvw));

  if numel(fnew) >= n
    sel_vw = fnz( ri (1:min (nnz, fnew (n))));
  else
    sel_vw = fnz(ri);
  end
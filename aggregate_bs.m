function [bs, uvw] = aggregate_bs (vw, bs, M_bin2compactbin, k)

	uvw = unique (vw);
	mid = zeros(k, 1);
	mid (uvw) = 1:numel (uvw);
	sub = mid (vw);			

	bs = aggregate_binary(bs, uint32 (sub), max (sub));

	% resolve conflicts
	con = (bs == 0.5);
	bs = (bs > 0.5);
	rnd	 = (rand( size(bs)) > 0.5);	% randomly resolve conflicts
	bs (con) = rnd (con);

	% re-compress signatures
	bs = M_bin2compactbin * bs;

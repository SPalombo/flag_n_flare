function [f,g] = softMaxFun(w, X, Y, lambdaL2)
%%
% @X is the (n x d) data matrix.
% @w is the (d x C) by 1 weight vector where C is the number of classes (% Technically the total number of classes is C+1, but the degree of freedom is only C).
% @Y is the (n x C) label matrix, i.e., Y(i,b) is one if i-th label is class
%    b, and other wise 0.

%% Example to generate data to test:
% n=10; d = 5; 
% total_no_of_C = 3; 
% X = randn(n,d); 
% I = eye(total_no_of_C,total_no_of_C-1); 
% ind = randsample(total_no_of_C,n,true); Y = I(ind, :); 
% w = randn(d*(total_no_of_C-1),1); 
% derivativeTest(@(w) softMaxFun(X,Y,w),randn(d*(total_no_of_C-1),1))

%%
%%%%%% Sanity Checks %%%%%%
assert(length(w) == length(shiftdim(w))); % to make sure w is a column vector
[n,d] = size(X);
assert(ceil(length(w)/d) == floor(length(w)/d)); % to make sure that w is of a vector of length (C x d) for some integer C, where C is the number of classes.
C = length(w)/d; % Technically the total number of classes is C+1, but the degree of freedom is only C
W = (vec2mat(w,d))'; % A (d x C) matrix formed from w where each column is a w_b for class b, b = 1,2,C
assert(size(W,1) == d);
assert(size(W,2) == C);
assert(size(Y,1) == n);
assert(size(Y,2) == C);
labels = sort( unique(sum(Y,2)) );
assert(length(labels) <= 2); % Y can only have zeros and ones
if length(labels) > 1
    assert( labels(1) == 0 );
    assert( labels(2) == 1);
else
    assert(labels == 0 || labels == 1);
end
assert(max(sum(Y,2)) >= 0 && max(sum(Y,2)) <=1); % each row of y has to have at most only one 1.
assert(sum(sum(Y,2) == ones(n,1)) >= 0 && sum(sum(Y,2) == ones(n,1))<=n); % each row of y has to have at most only one 1.


%%
%%%%% Function Value %%%%%
XW = X*W; % (n x C) matrix
% Do the Log-Sum-Trick...Our formulation is like "log ( 1 + sum_{c=1}^{C-1}
% exp<x_{i}, w_{c}> )"
large_vals = max(0,max(XW,[],2)); % ONLY WE CHECK FOR MAXIMUM POSTIVE VALUES!
XW_trick = XW - repmat(large_vals,1,C);
XW_1_trick = [-large_vals, XW_trick]; %To account for the "1" in the sum, i.e., exp(0)
sum_exp_trick = sum(exp(XW_1_trick),2);
log_sum_exp_trick = large_vals + log (sum_exp_trick); % (n x 1)
f = sum(log_sum_exp_trick) - sum(sum(XW.*Y,2)) + 0.5*lambdaL2*(norm(w))^2;

%%
%%%%% Gradient %%%%%
if nargout >= 2
    inv_sum_exp = 1./sum_exp_trick;
    inv_sum_exp = repmat(inv_sum_exp,1,C);
    S = inv_sum_exp.*exp(XW_trick); % (n x C) matrix
    assert(sum(sum((S <= 1),2)) == n*C); % every entry has to be smaller than 1!
    g_fun = (X')*(S - Y); %(d x C) matrix
    g_fun = g_fun(:); % gradient vector
    g = g_fun + lambdaL2*w;
end
end

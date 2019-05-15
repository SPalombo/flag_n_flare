function [ bound ] = calcBound( L, hist )
%
%   
numInst = size(hist.Betas, 2);
bound = zeros(numInst,1);
for i = 1:numInst
    num = hist.Betas(i)*L*hist.DInfSqr;
    bound(i) = num/(i^2);
end
end


function [ compFact ] = calcCompFact( hist )
%Given the history of an optimizer, computes the competative factor.
%   
compFact = (hist.DInfSqr/hist.D2Sqr)*hist.Betas;

end


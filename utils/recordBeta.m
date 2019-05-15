function hist = recordBeta( g, hist, iters, iter )
%Calc and Record Beta
fromStartVec = hist.xInit - iter;
currDInfSqr = norm(fromStartVec, inf)^2;
currD2Sqr = norm(fromStartVec, 2)^2;
if currDInfSqr > hist.DInfSqr
    hist.DInfSqr = currDInfSqr;
end
if currD2Sqr > hist.D2Sqr
    hist.D2Sqr = currD2Sqr;
end
d = size(g, 1);
hist.sqrGSum = hist.sqrGSum + g.^2;
Beta = (sum(sqrt(hist.sqrGSum)))^2/(iters * d);
hist.Betas = [hist.Betas, Beta]; 

end


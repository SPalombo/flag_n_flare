function [x,hist]=flare_alg(xinit, fun, prox, proj, L, maxX, testFun, xIsProx)
%% FLARE: FLAG RELaxed!
% Definition of "flare":
% fler/
% noun
% 1.
% a sudden brief burst of bright flame or light.
% "the flare of the match lit up his face"
% synonyms:	blaze, flash, dazzle, burst, flicker
% "the flare of the match"
% 2.
% a gradual widening, especially of a skirt or pants.
% "as you knit, add a flare or curve a hem"
%%
hist.objValue = [];
hist.nProxEvals = 0;
hist.sqrGSum = zeros(size(xinit));
hist.Betas = [];
hist.xInit = xinit;
hist.DInfSqr = 0;
hist.D2Sqr = 0;
doTestAccuracy = false;

overEstimateFactor = 1;
fprintf('....FLARE with Over-Estimate Factor %g\n', overEstimateFactor);
%iterates = zeros(size(xinit, 1), maxX + 1);
if exist('testFun','var') && ~isempty(testFun)
    doTestAccuracy = true;
    assert(isa(testFun,'function_handle'));
    hist.testAccuracy = [];
end
xk= xinit;
yk= xinit;
zk= xinit;
prox_xk = prox(xk);
xk_prev = xk;
prox_xk_prev = prox_xk;
iters = 1;
hist = recordHistory(hist, fun, testFun, yk, 1, maxX, doTestAccuracy, xIsProx);
stopCondMet = false;
Lk = 0;
Lk_last = 0;
etak = 0;
gksum=ones(size(xk)) * 1e-20;
sumetak=0;
k = 1;
while stopCondMet == false
    %take gradient step
    yk = prox_xk;
    pk = -L*(yk-xk);
    if norm(pk) == 0
        break;
    end
    gk = pk/norm(pk);
    gksum = gksum + gk.^2;
    sk = sqrt(gksum);
    
    %compute etak
    Lk_prev = Lk_last;
    Lk_last = Lk;
    Lk = L* sum(gk.^2./sk);
    if( k > 1 && Lk > Lk_last ) % If we did not use an overestimate of L_k in the last iterations
        overEstimateFactor = 1.1*overEstimateFactor;
        fprintf('Bad estimate for Lk at iterationb %g...Increasing over estimation factor to %g\n', k, overEstimateFactor);
        k = k-1;
        prox_xk = prox_xk_prev;
        xk = xk_prev;
        Lk = Lk_last;
        Lk_last = Lk_prev;
        hist.nProxEvals = hist.nProxEvals + 1;
        continue;
    else
        hist = recordBeta(gk, hist, iters, yk);         
        iters = iters + 1;
        hist = recordHistory(hist, fun, testFun, yk, 1 , maxX, doTestAccuracy, xIsProx);
        stopCondMet = checkStopCond(xIsProx, maxX, iters, hist);
        Lk = overEstimateFactor*Lk;
    end
    etak = 1/(2*Lk) + sqrt( (1/(4*Lk^2)) + ((etak^2)*Lk_last)/Lk);
    sumetak=sumetak + etak;
    
    %take mirror step
    zk = proj(zk - etak * pk./sk);
    
        xk_prev = xk;
        prox_xk_prev = prox_xk;
        wk = 1/(etak*Lk);
        xk = (1-wk)*yk + wk*zk;
        prox_xk = prox(xk);
    k = k+1;
end
x=yk;
end


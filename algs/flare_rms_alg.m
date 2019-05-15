function [x,hist, iterates]=flare_rms_alg(xinit, fun, prox, proj, flare_up, L, gamma, maxX, testFun, xIsProx)
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

if ~exist('flare_up','var')
    flare_up = true;
end
overEstimateFactor = 1;
if flare_up
    fprintf('....FLARING UP with Over-Estimate Factor %g\n', overEstimateFactor);
end
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
    gksum = gamma*gksum + (1-gamma)*gk.^2;
    sk = sqrt(gksum);
    
    %compute etak
    Lk_prev = Lk_last;
    Lk_last = Lk;
    Lk = L* sum(gk.^2./sk);
    if flare_up % We use the over-estimate of L_k 
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
%             if xIsProx == false
%                 iterates(:, iters) = yk;
%             end           
            iters = iters + 1;
            hist = recordHistory(hist, fun, testFun, yk, 1 , maxX, doTestAccuracy, xIsProx);
            stopCondMet = checkStopCond(xIsProx, maxX, iters, hist);
            Lk = overEstimateFactor*Lk;
        end
    end
    etak = 1/(2*Lk) + sqrt( (1/(4*Lk^2)) + ((etak^2)*Lk_last)/Lk);
    sumetak=sumetak + etak;
    
    %take mirror step
    zk = proj(zk - etak * pk./sk);
    
    %if flare_up
        xk_prev = xk;
        prox_xk_prev = prox_xk;
        wk = 1/(etak*Lk);
        xk = (1-wk)*yk + wk*zk;
        prox_xk = prox(xk);
%     else
%         % use "linearized" bisection to find xk from zk and yk
%         prox_xk = prox( yk );
%         prox_yk = prox_xk;
%         bisec_fun_1 = dot( prox_xk - ( yk  ), yk - zk);
%         if  bisec_fun_1 >= 0
%             xk = yk;
%             hist = recordHistory();
%             continue;
%         end
%         prox_xk = prox( zk );
%         prox_zk = prox_xk;
%         bisec_fun_0 = dot( prox_xk - ( zk  ), yk - zk);
%         if bisec_fun_0 <= 0
%             xk = zk;
%             hist = recordHistory(hist, fun, testFun, yk, 2, maxX, xIsProx, 1);
%             continue;
%         end
%         assert(bisec_fun_1 < 0);
%         assert(bisec_fun_0 > 0);
%         t = dot( zk-prox_zk , yk-zk )  / dot( prox_yk - yk + zk - prox_zk , yk-zk );
%         %%% Sanity check for t %%%
%         % bisec_fun = @(t) dot( ( t* prox_yk + (1-t)*prox_zk ) - ( t*yk + (1-t)*zk ), yk - zk );
%         % assert(abs(bisec_fun(t)) <= eps);
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%
%         xk = t*yk + (1-t)*zk;
%         prox_xk = prox(xk);
%         hist = recordHistory(hist, fun, testFun, yk, 3 , doTestAccuracy, 1);
%     end
    k = k+1;
end
x=yk;
% if xIsProx == false
%     iterates(:, end) = x;
% end
end
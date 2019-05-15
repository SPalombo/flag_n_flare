function [x,hist]=flare_alg(xinit, fun, prox, flare_up, L, maxIter, maxProxEvals, testFun)
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
hist.objValueProx = [];
hist.objValueItrs = [];
hist.nProxEvals = 0;
doTestAccuracy = false;

if ~exist('flare_up','var') 
    flare_up = true;
end
if flare_up 
    fprintf('...... FLARING UP .....');
end
if exist('testFun','var') && ~isempty(testFun)
    doTestAccuracy = true;
    assert(isa(testFun,'function_handle'));
    hist.testAccuracyProx = [];
    hist.testAccuracyItrs = [];
end
xk= xinit;
yk= xinit;
zk= xinit;
prox_xk = prox(xk);
hist = recordHistory(hist, fun, testFun, yk, 1, doTestAccuracy);
Lk = 0;
etak = 0;
gksum=ones(size(xk)) * 1e-20;
sumetak=0;
for k = 1:maxIter
    if hist.nProxEvals > maxProxEvals
        break;
    end
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
    Lk_last = Lk;
    Lk = L* sum(gk.^2./sk);
    etak = 1/(2*Lk) + sqrt( (1/(4*Lk^2)) + ((etak^2)*Lk_last)/Lk);
    sumetak=sumetak + etak;
    
    %take mirror step
    zk = zk - etak * pk./sk;
    
    if flare_up
        wk1 = sqrt(1/k);
        xk = (1-wk1)*yk + wk1*zk;
        fun1 = fun(xk);
        if fun1 > hist.objValueProx(end)
            wk = 1/k;
            xk = (1-wk)*yk + wk*zk;
        end
        prox_xk = prox(xk);
        hist = recordHistory(hist, fun, testFun, yk, 1 , doTestAccuracy);
    else
        % use bisection to find xk from zk and yk
        prox_xk = prox( yk );
        prox_yk = prox_xk;
        bisec_fun_1 = dot( prox_xk - ( yk  ), yk - zk);
        if  bisec_fun_1 >= 0
            xk = yk;
            hist = recordHistory(hist, fun, testFun, yk, 1, doTestAccuracy);
            continue;
        end
        prox_xk = prox( zk );
        prox_zk = prox_xk;
        bisec_fun_0 = dot( prox_xk - ( zk  ), yk - zk);
        if bisec_fun_0 <= 0
            xk = zk;
            hist = recordHistory(hist, fun, testFun, yk, 2, doTestAccuracy);
            continue;
        end
        assert(bisec_fun_1 < 0);
        assert(bisec_fun_0 > 0);
        t = dot( zk-prox_zk , yk-zk )  / dot( prox_yk - yk + zk - prox_zk , yk-zk );
        %%% Sanity check for t %%%
        % bisec_fun = @(t) dot( ( t* prox_yk + (1-t)*prox_zk ) - ( t*yk + (1-t)*zk ), yk - zk );
        % assert(abs(bisec_fun(t)) <= eps);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        xk = t*yk + (1-t)*zk;
        prox_xk = prox(xk);
        hist = recordHistory(hist, fun, testFun, yk, 3 , doTestAccuracy);
    end
end
x=yk;
end


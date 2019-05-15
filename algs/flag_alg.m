%algorithm for optimizing f(x) given by argument f. L is the lipschitz
%constant,
%L: lipschitz constant of gradient, used to determine stepsize of yk
%iter: total number of iterations to run algorithm for
%xinit: initial x value
%f: function object
%prox: ignore this

%x: returned final solution, x=yk for k=iter
%hist: hist of f(yk), used for plotting purposes
%improvementfactor: ignore this
function [x,hist]=flag_alg(xinit, fun, prox, proj, bisectEpsilon, L, maxX, testFun, xIsProx)
hist.objValue = [];
hist.nProxEvals = 0;
hist.sqrGSum = zeros(size(xinit));
hist.Betas = [];
hist.xInit = xinit;
hist.DInfSqr = 0;
hist.D2Sqr = 0;
doTestAccuracy = false;
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
iters = 1;
hist = recordHistory(hist, fun, testFun, yk, 1, maxX, doTestAccuracy, xIsProx);
stopCondMet = false;
Lk = 0;
etak = 0;
gksum=ones(size(xk)) * 1e-20;
sumetak=0;
while stopCondMet == false
%     if xIsProx == false
%         iterates(:, iters) = yk;
%     end

    iters = iters + 1;
    %take gradient step
    yk = prox_xk; 
    pk = -L*(yk-xk);
    if norm(pk) == 0
        break;
    end
    gk = pk/norm(pk);
    gksum=gksum+gk.^2;
    sk = sqrt(gksum);
    hist = recordBeta(gk, hist, iters, yk);
    
    %compute etak
    Lk_last = Lk;
    Lk = L* sum(gk.^2./sk);
    etak = 1/(2*Lk) + sqrt( (1/(4*Lk^2)) + ((etak^2)*Lk_last)/Lk);
    sumetak=sumetak + etak;
    
    %take mirror step
    zk = proj(zk - etak * pk./sk);
    
    %use Binary search to find xk from zk and yk
    prox_xk = prox( yk );
    bisec_fun_1 = dot( prox_xk - ( yk  ), yk - zk);
    if  bisec_fun_1 >= 0
        xk = yk;
        hist = recordHistory(hist, fun, testFun, yk, 1, maxX, doTestAccuracy, xIsProx);
        stopCondMet = checkStopCond( xIsProx, maxX, iters, hist );
        continue;
    end
    prox_xk = prox( zk );
    bisec_fun_0 = dot( prox_xk - ( zk  ), yk - zk);
    if bisec_fun_0 <= 0
        xk = zk;
        hist = recordHistory(hist, fun, testFun, yk, 2, maxX, doTestAccuracy, xIsProx);
        stopCondMet = checkStopCond( xIsProx, maxX, iters, hist );
        continue;
    end
    % we now enter the bisection
    assert(bisec_fun_1 < 0);
    assert(bisec_fun_0 > 0);
    bisec_fun = @(t) dot( prox( t*yk + (1-t)*zk )- ( t*yk + (1-t)*zk ), yk - zk );
    [t,nBisections] = bisect( bisec_fun, 0, 1, bisec_fun_0, bisec_fun_1, bisectEpsilon);
    xk = t*yk + (1-t)*zk;
    prox_xk = prox(xk);
    hist = recordHistory(hist, fun, testFun, yk, 3 + nBisections, maxX, doTestAccuracy, xIsProx);
    stopCondMet = checkStopCond( xIsProx, maxX, iters, hist );
end
x=yk;
% if xIsProx == false
%     iterates(:, end) = x;
% end
end


function [x,hist, iterates]=fista_alg(xinit, fun, prox, maxX, testFun, xIsProx)
hist.objValue = [];
hist.nProxEvals = 0;
doTestAccuracy = false;
%iterates = zeros(size(xinit, 1), maxX + 1);
if exist('testFun','var') && ~isempty(testFun)
    doTestAccuracy = true;
    assert(isa(testFun,'function_handle'));
    hist.testAccuracy = [];
end
xk=xinit;
hist = recordHistory(hist, fun, testFun, xk, 1, maxX, doTestAccuracy, xIsProx);
yk=xinit;
lambdak=0;
for k = 1:maxX
%     if xIsProx == false
%         iterates(:, k) = xk;
%     end
    lambdako = lambdak;
    lambdak = (1+sqrt(1+4*lambdako^2))/2;
    lambdakn = (1+sqrt(1+4*lambdak^2))/2;
    gammak = (1-lambdak)/lambdakn;
    ykn = prox(xk);
    xkn = (1-gammak)*ykn + gammak * yk;
    yk=ykn;
    xk=xkn;
    hist = recordHistory(hist, fun, testFun, yk, 1, maxX, doTestAccuracy, xIsProx);
end
x=xk;
% if xIsProx == false
%     iterates(:, end) = x;
% end
end


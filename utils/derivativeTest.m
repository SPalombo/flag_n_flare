function derivativeTest(fun,x0)
[f0,g0] = fun(x0);
dx = randn(size(x0));
M = 20;
dxs = zeros(M,1);
firstOrderError = zeros(M,1);
for i = 1:M
    x = x0 + dx;
    [f,~] = fun(x);
    firstOrderError(i) = abs(f - ( f0 + (dx')*g0 ))/f0;
    fprintf('First Order Error: %g\n',firstOrderError(i));
    dxs(i) = norm(dx);
    dx = dx/2;
end

figure(1);
semilogy(1:M,abs(firstOrderError),'r',1:M,dxs.^2,'b');
legend('1st Order Error','order');

end
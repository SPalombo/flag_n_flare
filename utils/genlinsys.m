%returns A and b that encode the function f(x) = 1/2 x^T Ax -b^T x
function [A,b,theta]=genlinsys(n, d, c)

A = randn(n,d);
A(:,1:end-10) = A(:,1:end-10)*c;
theta = randn(d,1);
b = A*theta + (1E-5)*normrnd(0, 0.1);
totalData = [A,b];
csvwrite('data/genSys',totalData);
end




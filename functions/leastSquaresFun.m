function [f,g, H] = leastSquaresFun(x, A, b, lambdaL2)
n = size(A,1);
p = size(A,2);
assert(n == length(b));
assert(p == length(x));

Ax = A*x;
f = 0.5*norm(Ax - b)^2 + 0.5*lambdaL2*(norm(x))^2;
if nargout >= 2
    g = A'* (Ax - b) + lambdaL2*x;
end

if nargout >= 2
    H = A'* (Ax - b) + lambdaL2*x;
end

end

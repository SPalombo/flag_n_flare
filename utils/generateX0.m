function [ x0 ] = generateX0( N )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
s = round(N*.03);
rand('state', 1);
randn('state', 1);
sel = randperm(N); sel = sel(1:s);
x0 = zeros(N,1); x0(sel) = 1;
x0 = x0 .* sign(randn(N,1)) .* (1-.3*rand(N,1));
end


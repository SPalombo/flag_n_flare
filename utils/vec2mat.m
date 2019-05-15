function [ mat ] = vec2mat( vec, N )
%VEC2MAT builds res-matrix from res-vector
%   mat:  ceil(M) columns, N rows
%   vec:  has to be row vector

segs=length(vec);
M=segs/N;
Mc=ceil(M);
vec=[vec nan(  1, Mc*N-segs)];

mat=reshape(vec,N,Mc).';


end


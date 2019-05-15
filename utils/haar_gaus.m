function [ haar_vec ] = haar_gaus(img_vec, gauss_blur, img_size, level, max_level)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
img = reshape(img_vec, [img_size, img_size]);
blurred = gauss_blur(img);
jmin = max_level - level;
haar_trans = perform_haar_transf(blurred, jmin, +1);
% haar_trans = perform_wavelet_transf(blurred, jmin, +1);
haar_vec = haar_trans(:);
% haar_vec = reshape(haar_trans, [img_size^2, 1]);
end



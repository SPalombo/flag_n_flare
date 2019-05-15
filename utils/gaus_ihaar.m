function [ img_vec ] = gaus_ihaar( haar_vec, gauss_blur, img_size, level, max_level)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
haar_mat = reshape(haar_vec, [img_size, img_size]);
jmin = max_level - level;
pre_blur = perform_haar_transf(haar_mat, jmin, -1);
% pre_blur = perform_wavelet_transf(haar_mat, jmin, -1);
img =  gauss_blur(pre_blur);
img_vec = img(:);
% img_vec = reshape(img, [img_size^2, 1]);
end



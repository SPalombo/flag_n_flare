
function [ f, g] = gaus_blur_on_wavelet(A, A_t, x, b)

%make gaussian filter

f = 0.5*norm(A(x) - b)^2;

if nargout == 2
   g  = A_t(A(x) - b);

if nargout == 2
    
end
end
end

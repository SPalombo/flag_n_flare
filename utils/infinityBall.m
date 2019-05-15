function xx = infinityBall(x,radius)
assert(radius >= 0);
xx = shiftdim(x);
I = x > radius;
xx(I) = radius;
I = x < -radius;
xx(I) = -radius;
end
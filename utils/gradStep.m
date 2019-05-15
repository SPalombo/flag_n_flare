function v = gradStep(x,L,f)
[~,g] = f(x);
v = x - g/L;
end
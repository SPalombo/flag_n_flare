function [x,hist] = fista_st_con( L, iter, xinit, f, prox, muif, muig)

mui = muig + muif;
q = (1/L)*mui/(1 + (1/L)*muig);
q = 0;
hist = zeros(iter,1);
xk=xinit;
yk=xinit;
lambdak=0;
lambdake = 0;
for i = 1:iter
    
    lambdako = lambdak;
    lambdak = (1-q*lambdako^2 + sqrt((1 - q*lambdako^2)^2 + 4*lambdako^2))/2;    
    lambdakn = (1-q*lambdak^2 + sqrt((1 - q*lambdak^2)^2 + 4*lambdak^2))/2;  
    gammak = (1-lambdak)/lambdakn * ((1 + (1/L)*muig - lambdakn*(1/L)*mui)/(1 - (1/L)*muif));
    
    ykn = prox(xk-1/L * (gradient(xk)));
    xkn = (1-gammak)*ykn + gammak * yk;
    
    yk=ykn;
    xk=xkn;
    
    hist(i) = eval(xk);

end
x=xk;

function [grad]= gradient(v)
[~,grad] = f(v);
end

function val = eval(v)
[val,~] = f(v);
end

end



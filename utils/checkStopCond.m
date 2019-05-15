function [ stopCondMet ] = checkStopCond( xIsProx, maxX, iters, hist )
stopCondMet = false;
if xIsProx
    if hist.nProxEvals > maxX
        stopCondMet = true;
    end
else
    if iters > maxX
        stopCondMet = true;
    end
end


end


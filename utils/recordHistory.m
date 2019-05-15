function hist = recordHistory(hist, fun, testFun, x, nProxEvals, maxX, doTestAccuracy, xIsProx)
%%%%%%%%%%%%% record obj, test, and beta %%%%%%%%%%%%%%%%%%% 

%calc range
endIndex = length(hist.objValue) +1;
if xIsProx
    startInd = endIndex;
    endInd = min(endIndex + nProxEvals, maxX);
    range = startInd:endInd;
else
    range = endIndex;
end
hist.nProxEvals = hist.nProxEvals + nProxEvals;

%Calc and Record Obj
hist.objValue(range) = fun(x);

%Calc and Record Test Accuracy
if doTestAccuracy
    hist.testAccuracy(range) = testFun(x);
end


end

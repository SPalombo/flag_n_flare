function setup_and_run(dataName, subProblemType, xMax, initVec, xAxisUnits)
if strcmp(dataName, 'blogdata') || ...
        strcmp(dataName, 'power-plant') || ...
        strcmp(dataName, 'news-populairty') || ...
        strcmp(dataName, 'housing') || ...
        strcmp(dataName, 'BlogFeedback') || ...
        strcmp(dataName, 'forest-fire') || ...
        strcmp(dataName, 'Facebook CVD') || ...
        strcmp(dataName, 'UJIIndoorLoc-regression') || ...
        strcmp(dataName, 'music')
    isRegression = true;
    isClassification = false;
    fig_test_title = 'Test Error';
    fprintf('---------------------- REGRESSION -----------------------\n\n');
end
if  strcmp(dataName,'arcene') || ...
        strcmp(dataName,'dorothea') || ...
        strcmp(dataName,'Twenty Newsgroups') || ...
        strcmp(dataName,'Forest Covertype') || ...
        strcmp(dataName, 'mnist') || ...
        strcmp(dataName, 'UJIIndoorLoc-classification') || ...
        strcmp(dataName, 'Gisette') || ...
        strcmp(dataName, 'HARUS')
    isRegression = false;
    isClassification = true;
    fig_test_title = 'Test Accuracy';
    fprintf('-------------------- CLASSIFICATION ----------------------\n\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Loading Data %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[A_train, b_train, A_test, b_test] = loadData(dataName);
n_train = size(A_train,1);
n_test = size(A_test,1);
d = size(A_train,2);
b_train = shiftdim(b_train);
assert(length(b_train) == n_train);
nClasses = size(b_train,2)+1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% End Loading Data %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Beging Parameters %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%% Regularization %%%%%%%%%%%%%%%%%%%%%
assert(strcmp( subProblemType, 'Box_Constrained') || ...
       strcmp( subProblemType, 'L1_Regularized') || ...
       strcmp( subProblemType, 'L2_Regularized'));
if strcmp(subProblemType, 'L1_Regularized')
    assert(strcmp(subProblemType, 'Box_Constrained') == false);
    assert(strcmp(subProblemType, 'L2_Regularized') == false);
    lambdaL1 = 0.1;
    lambdaL2 = 0;
    radius = inf;
elseif strcmp(subProblemType, 'L2_Regularized')
    assert(strcmp(subProblemType, 'Box_Constrained') == false);
    lambdaL1 = 0;
    lambdaL2 = 0.1;
    radius = inf;
else
    assert(strcmp(subProblemType, 'L1_Regularized') == false);
    lambdaL1 = 0;
    lambdaL2 = 0;
    radius = 1;
end
lambda = max(lambdaL1, lambdaL2);

%%%%%%%%%%%%%% Biesction Epsilon %%%%%%%%%%%%%%%%%
bisectEpsilon = 0.001;


%%%%%%%%%%%%%%%%% Initial Vec %%%%%%%%%%%%%%%%%%%%
assert(strcmp( initVec, 'Zero') || ...
       strcmp( initVec, 'One'))
if strcmp(initVec, 'Zero')
    assert(strcmp(initVec, 'One') == false);
    x0 = zeros(d*(nClasses-1),1);
else
    assert(strcmp(initVec, 'Zero') == false);
    x0 = ones(d*(nClasses-1),1);
    x0 = x0/norm(x0);
end

%%%%%%%%%%%%%%%%%%% X Axis %%%%%%%%%%%%%%%%%%%%%%%%% 
assert(strcmp( xAxisUnits, 'Iterations') || ...
       strcmp( xAxisUnits, 'Prox Evaluations'));
xIsProx = strcmp(xAxisUnits, 'Prox Evaluations');

%%%%%%%%%%%%%%%%% Verificaton that Data Dims are proper %%%%%%%%%%%%%%
assert( length(x0) == ( size(A_train,2)*size(b_train,2) ) );
if isClassification
    assert( length(x0) == ( size(A_train,2)*( size(b_test,2)-1 ) ) );
else
    assert( length(x0) == ( size(A_train,2)* size(b_test,2)  ) );
end
fprintf('\nData: %s, n_train: %g, n_test: %g, d: %g, C: %g\n',dataName, n_train , n_test, d , nClasses);
fprintf('\nReg Param: %g, Inf-Ball Radius: %g, FLAG Bisect: %g, X Axis Units: %s, Max X Units: %g, Norm(x0): %g\n', lambda, radius, bisectEpsilon, xAxisUnits, xMax, norm(x0));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% End Parameters %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if isClassification
    %%%%%%%%%%%%%%%%% BEGIN FUNCTIONS FOR CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%
    assert(isRegression == false);
    lossFun = @(x) softMaxFun(x, A_train, b_train, lambdaL2);
    regFun = @(x) regularizerFunL1(x,lambdaL1);
    objFun = @(x) lossFun(x) + regFun(x);
    L = 0.25*(svds(A_train,1))^2;
    testFun = @(x) testClassificationResult(A_test, b_test, x);
    %%%%%%%%%%%%%%%%% END FUNCTIONS FOR CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%
else
    %%%%%%%%%%%%%%%%% BEGIN FUNCTIONS FOR REGRESSION %%%%%%%%%%%%%%%%%%%%%%%%%
    assert(isRegression == true);
    assert(isClassification == false);
    lossFun = @(x) leastSquaresFun(x, A_train, b_train, lambdaL2);
    regFun = @(x) regularizerFunL1(x,lambdaL1);
    objFun = @(x) lossFun(x) + regFun(x);
    L = norm(full(A_train'*A_train));
    testFun = @(x) testRegressionResult(A_test, b_test, x);
    %%%%%%%%%%%%%%%%% END FUNCTIONS FOR REGRESSION %%%%%%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%Begining of Prox and Projection Function Handle %%%%%%%%%%%%%%%%%%%%%
projFun = @(x) infinityBall( x, radius );
proxFun = @(v) projFun( perform_thresholding( gradStep( v,L,lossFun ), lambdaL1/L, 'soft') );
%%%%%%%%%%%%%%%% End of  Prox and Projection Function Handle %%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%% Run FLAG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nRunning FLAG...............');
tic;
[x_flag, hist_flag]= flag_alg(x0, objFun, proxFun, projFun, bisectEpsilon, L, xMax, testFun, xIsProx);
tElapsed = toc;
fprintf('Done in %g secs.\n',tElapsed);
if isClassification
    assert(isRegression == false);
    [accuracy_flag, prob_flag, class_assignment_flag] = testFun(x_flag);
    nnz_flag = sum(abs(x_flag) >= 1E-12);
    fprintf('FLAG test accuracy %g with total prox evaluations of %g and nnz%%: %g.\n',accuracy_flag, hist_flag.nProxEvals, 100*nnz_flag/(d*(nClasses-1)));
end
if isRegression
    assert(isClassification == false);
    accuracy_flag = testFun(x_flag);
    fprintf('FLAG test accuracy %g with total prox evaluations of %g.\n',accuracy_flag, hist_flag.nProxEvals);
end
%%%%%%%%%%%%%%%%% End FLAG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% Run FLARE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nRunning FLARE with "flare_up" option..........');
flare_up = true;
%flareMaxIters = hist_flag.nProxEvals;
tic;
[x_flare_up, hist_flare_up]= flare_alg(x0, objFun, proxFun, projFun, L, xMax, testFun, xIsProx);
tElapsed = toc;
fprintf('Done in %g secs.\n',tElapsed);
if isClassification
    assert(isRegression == false);
    [accuracy_flare_up, prob_flare_up, class_assignment_flare_up] = testFun(x_flare_up);
    nnz_flare_up = sum(abs(x_flare_up) >= 1E-12);
    fprintf('FLARE UP test accuracy %g with total prox evaluations of %g and nnz%%: %g.\n',accuracy_flare_up, hist_flare_up.nProxEvals, 100*nnz_flare_up/(d*(nClasses-1)));
end
if isRegression
    assert(isClassification == false);
    accuracy_flare_up = testFun(x_flare_up);
    fprintf('FLARE UP test accuracy %g with total prox evaluations of %g.\n',accuracy_flare_up, hist_flare_up.nProxEvals);
end
%%%%%%%%%%%%%%%%% End FLARE XXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%% Run FISTA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nRunning FISTA...............');
%fistaMaxIters = hist_flag.nProxEvals;
tic;
[x_fista, hist_fista]= fista_alg( x0, objFun, proxFun, xMax, testFun, xIsProx);
tElapsed = toc;
fprintf('Done in %g secs.\n',tElapsed);

if isClassification
    assert(isRegression == false);
    [accuracy_fista, prob_fista, class_assignment_fista] = testFun(x_fista);
    nnz_fista = sum(abs(x_fista) >= 1E-12);
    fprintf('FISTA test accuracy %g with total prox evaluations of %g and nnz%%: %g.\n',accuracy_fista, hist_flag.nProxEvals, 100*nnz_fista/(d*(nClasses-1)));
end
if isRegression
    assert(isClassification == false);
    accuracy_fista = testFun(x_fista);
    fprintf('FISTA test accuracy %g with total prox evaluations of %g.\n',accuracy_fista, hist_flag.nProxEvals);
end
%%%%%%%%%%%%%%%%% End FISTA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Data Capture %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%% Producing Folder %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(xAxisUnits, 'Iterations')
    xAxisFName = 'iters';
else
    xAxisFName = 'prox';
end
dir_name = ['./results/',dataName, '_', xAxisFName, '_', subProblemType];
if ~exist(dir_name, 'dir')
    status = mkdir(dir_name);
end
file_name = [dir_name,'/',dataName];

% load(file_name);

%%%%%%%%%%%%%%%%%%%%%%%%%% Making Plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% Writting Iterate Files %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% csvwrite([file_name, 'flag_iterates.csv'], flag_iterates);
% csvwrite([file_name, 'flare_iterates.csv'], flare_iterates);
% csvwrite([file_name, 'fista_iterates.csv'], fista_iterates);


%%%%%%%%%%%%%%%%%%%%%%%%%% Objective Value Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
linewidth = 2;
figure(1);
semilogy(1:length(hist_fista.objValue), hist_fista.objValue, 'b', 'LineWidth',linewidth);
hold on;
semilogy(1:length(hist_flag.objValue),hist_flag.objValue,'r','LineWidth',linewidth);
%semilogy(1:length(hist_flare.objValueProx),hist_flare.objValueProx,'k','LineWidth',linewidth);
semilogy(1:length(hist_flare_up.objValue),hist_flare_up.objValue,'color', [0 0.5 0],'LineWidth',linewidth);
legend('FISTA','FLAG', 'FLARE');
title(['Objective Function: ', dataName]);
xlabel(['No. of ', xAxisUnits]);
saveas(gcf,[file_name,'Obj'],'fig');
saveas(gcf,[file_name,'Obj'],'png');


%%%%%%%%%%%%%%%%%%%%%%%%%% Test Error Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
if isRegression
    semilogy(1:length(hist_fista.testAccuracy), hist_fista.testAccuracy,'b','LineWidth',linewidth);
    hold on;
    semilogy(1:length(hist_flag.testAccuracy),hist_flag.testAccuracy,'r','LineWidth',linewidth);
    %semilogy(1:length(hist_flare.testAccuracyProx),hist_flare.testAccuracyProx,'k','LineWidth',linewidth);
    semilogy(1:length(hist_flare_up.testAccuracy),hist_flare_up.testAccuracy,'color', [0 0.5 0],'LineWidth',linewidth);
else
    plot(1:length(hist_fista.testAccuracy), hist_fista.testAccuracy,'b','LineWidth',linewidth);
    hold on;
    plot(1:length(hist_flag.testAccuracy),hist_flag.testAccuracy,'r','LineWidth',linewidth);
    %plot(1:length(hist_flare.testAccuracyProx),hist_flare.testAccuracyProx,'k','LineWidth',linewidth);
    plot(1:length(hist_flare_up.testAccuracy),hist_flare_up.testAccuracy,'color', [0 0.5 0],'LineWidth',linewidth);
end
legend('FISTA','FLAG', 'FLARE','Location', 'East');
title([fig_test_title,': ', dataName]);
xlabel(['No. of ', xAxisUnits]);
saveas(gcf,[file_name,'Accu'],'fig');
saveas(gcf,[file_name,'Accu'],'png');




%%%%%%%%%%%%%%%%%%%%%%%%%% Beta Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(3);
plot(1:length(hist_flag.Betas), hist_flag.Betas, 'r','LineWidth',linewidth);
hold on;
plot(1:length(hist_flare_up.Betas), hist_flare_up.Betas, 'color', [0 0.5 0],'LineWidth',linewidth);
legend('FLAG', 'FLARE','Location', 'East');
title(['Beta Value: ', dataName]);
xlabel(['No. of ', xAxisUnits]);
saveas(gcf,[file_name,'Beta'],'fig');
saveas(gcf,[file_name,'Beta'],'png');

autoArrangeFigures(2,2,1);

save(file_name,'x_fista', 'hist_fista', ...
    'x_flag', 'hist_flag', ...
    'x_flare_up','hist_flare_up', ...
    'lambda', ...
    'bisectEpsilon', ...
    'xAxisUnits', ...
    'xMax', ...
    'x0', ...
    'subProblemType', ...
    'initVec');

fclose('all');
end
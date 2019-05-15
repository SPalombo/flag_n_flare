clc;clear;close all;
addpath(genpath(pwd));
rng('default');
rng(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Regression Data %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dataName = 'blogdata';
% dataName = 'power-plant';
% dataName = 'news-populairty';
% dataName = 'housing';
% dataName = 'BlogFeedback';
% dataName = 'forest-fire';
% dataName = 'UJIIndoorLoc-regression';
% dataName = 'Facebook CVD';
% dataName = 'music';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Classification Data %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dataName = 'arcene'; % Binary
% dataName = 'dorothea'; % Binary
% dataName = 'Twenty Newsgroups'; % Multi-Class
% dataName = 'Forest Covertype'; % Multi-Class
% dataName = 'mnist'; % Multi-Class
% dataName = 'UJIIndoorLoc-classification';
dataName = 'Gisette';
% dataName = 'HARUS';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Sub-Problem Type %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 subProblemType = 'L1_Regularized';
% subProblemType = 'L2_Regularized';
% subProblemType = 'Box_Constrained';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Initialization Vector %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 initVec = 'Zero';
% initVec = 'One';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% X-Axis %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% xAxis = 'Iterations';
 xAxis = 'Prox Evaluations';

xMax = 999;

setup_and_run(dataName, subProblemType, xMax, initVec, xAxis);


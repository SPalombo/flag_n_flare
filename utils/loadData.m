function [A_train, b_train, A_test, b_test] = loadData(dataName)

fprintf('..................LOADING DATA.................\n');

assert(strcmp(dataName,'arcene') || ... % start of classification names
    strcmp(dataName,'dorothea') || ...
    strcmp(dataName,'genSysReg') || ...
    strcmp(dataName,'genSysClass') || ...
    strcmp(dataName,'Twenty Newsgroups') || ...
    strcmp(dataName,'Forest Covertype') || ...
    strcmp(dataName, 'mnist') || ... 
    strcmp(dataName, 'UJIIndoorLoc-classification') || ... 
    strcmp(dataName, 'Gisette') || ...
    strcmp(dataName, 'HARUS') || ...% end of classification names
    strcmp(dataName, 'blogdata') ||... % start of regression names
    strcmp(dataName, 'power-plant') || ...
    strcmp(dataName, 'news-populairty') || ...
    strcmp(dataName, 'housing') || ...
    strcmp(dataName, 'BlogFeedback') || ...
    strcmp(dataName, 'forest-fire') || ...
    strcmp(dataName, 'Facebook CVD') || ...
    strcmp(dataName, 'UJIIndoorLoc-regression') || ...
    strcmp(dataName, 'music')); % end of regression names


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% BEGIN CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%% Beging Arcene: Binary Classificaion  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(dataName, 'arcene')
    A_train = importdata('./data/arcene/arcene_train.data');
    labels_train_raw = importdata('./data/arcene/arcene_train.labels');
    b_train = getClassLabels(labels_train_raw, true);
    A_test = importdata('./data/arcene/arcene_valid.data');
    labels_test_raw= importdata('./data/arcene/arcene_valid.labels');
    b_test = getClassLabels(labels_test_raw, false);
end
%%%%%%%%%%%%% End Arcene %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% Begin Dorothea: Binary Classificaion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'dorothea')
    load('./data/dorothea/dorothea_train.mat');
    A_train = doroth_train_data;
    b_train = getClassLabels(doroth_train_labels, true);
    load('./data/dorothea/dorothea_valid.mat');
    A_test = doroth_valid_data;
    b_test = getClassLabels(doroth_valid_labels, false);
end
%%%%%%%%% End Dorothea %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% Beging 20news: Multi-Class Classificaion  %%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'Twenty Newsgroups')
    A_raw = load('./data/news/train.data');
    A_raw = sparse(A_raw(:,1),A_raw(:,2),A_raw(:,3));
    labels_raw = load('./data/news/train.label');
    b_raw = getClassLabels(labels_raw, false);
    test_index = randSample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw(train_index,:);
    b_train = b_raw(train_index,:);
    b_train = b_train(:,1:end-1);
    A_test = A_raw(test_index,:);
    b_test = b_raw(test_index,:);
end
%%%%%%%%% End 20news %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% Begin CoveType: Multi-Class Classificaion %%%%%%%%%%%%%
if strcmp(dataName, 'Forest Covertype')
    DD_train = importdata('./data/CoveType/covtype.data');
    A_raw = DD_train( :, 1:54 );
    labels_raw = DD_train( :, 55 );
    b_raw = getClassLabels(labels_raw, false);
    test_index = randSample(size(A_raw,1), ceil(0.25*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    b_train = b_train(:,1:end-1);
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end
%%%%%%%%%%%%%% End CoveType %%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'mnist')
    A_train = loadMNISTImages('./data/MNIST/train-images-idx3-ubyte');
    A_train = A_train';
    labels_train_raw = loadMNISTLabels('./data/MNIST/train-labels-idx1-ubyte');
    b_train = getClassLabels(labels_train_raw, true);
    A_test = loadMNISTImages('./data/MNIST/t10k-images-idx3-ubyte');
    A_test = A_test';
    labels_test_raw = loadMNISTLabels('./data/MNIST/t10k-labels-idx1-ubyte');
    b_test = getClassLabels(labels_test_raw, false);    
end

% %%%%%%%%% UJIIndoorLoc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% http://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc
% Attribute 523 (Floor)
% Attribute 524 (BuildingID)
if strcmp(dataName, 'UJIIndoorLoc-classification')
    attr = 523;
    DD_train = xlsread('./data/UJIndoorLoc/trainingData.csv');
    A_train = DD_train( :, 1:520 );
    labels_train_raw = DD_train(:, attr);
    b_train = getClassLabels(labels_train_raw, true);
    DD_test = xlsread('./data/UJIndoorLoc/validationData.csv');
    A_test = DD_test( : , 1:520 );
    labels_test_raw = DD_test( : , attr );
    b_test = getClassLabels(labels_test_raw, false);
end

%%%%%%%%% Beging gisette: Multi-Class Classificaion  %%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'Gisette')
    A_train = load('./data/gisette/gisette_train.data');
    labels_train_raw = load('./data/gisette/gisette_train.labels');
    b_train = getClassLabels(labels_train_raw, true);
    A_test = load('./data/gisette/gisette_valid.data');
    labels_test_raw = load('./data/gisette/gisette_valid.labels');
    b_test = getClassLabels(labels_test_raw , false);
end

%%%%%%%%% Beging gisette: Multi-Class Classificaion  %%%%%%%%%%%%%%%%%%%%%%
%% http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
if strcmp(dataName, 'HARUS')
    A_train = importdata('./data/HAPT/Train/X_train.txt');
%     A_train = normalizeData(A_train);
    labels_train_raw = load('./data/HAPT/Train/y_train.txt');
    b_train = getClassLabels(labels_train_raw, true);
    A_test = load('./data/HAPT/Test/X_test.txt');
%     A_test = normalizeData(A_test);
    labels_test_raw = load('./data/HAPT/Test/y_test.txt');
    b_test = getClassLabels(labels_test_raw , false);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% END CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% BEGIN RERGRESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% Begin facebook data: Regression %%%%%%%%%%%%%
if strcmp(dataName, 'Facebook CVD')
    DD_train = csvread('./data/facebook/Features_Variant_1.csv');
    A_raw = DD_train( :, 1:53 );
    b_raw = DD_train( :, 54 );
    test_index = randSample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% Power Plant %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% Begin blog data: Regression %%%%%%%%%%%%%
if strcmp(dataName, 'blogdata')
    DD_train = csvread('./data/sample_data/blogData_test-2012.03.20.00_00.csv');
    A_raw = DD_train( :, 1:50 );
    b_raw = DD_train( :, 51 );
    test_index = randSample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% Power Plant %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'power-plant')
    DD_train = csvread('./data/CCPP/power_plant.csv');
    A_raw = DD_train( :, 1:4 );
    b_raw = DD_train(:, 5);
    test_index = randSample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% News Popularity %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'news-populairty')
    DD_train = csvread('./data/OnlineNewsPopularity/OnlineNewsPopularity_no_url.csv');
    A_raw = DD_train( :, 1:59 );
    b_raw = DD_train(:, 60);
    test_index = randSample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% Housing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'housing')
    DD_train = csvread('./data/housing/housing.csv');
    A_raw = DD_train( :, 1:13 );
    b_raw = DD_train(:, 14);
    test_index = randSample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% Blog Feedback %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'BlogFeedback')
    DD_train = csvread('./data/BlogFeedback/blogData_train.csv');
    A_raw = DD_train( :, 1:280 );
    b_raw = DD_train(:, 281);
    test_index = randSample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% Forest Fire %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'forest-fire')
    DD_train = csvread('./data/forest_fire/forestfires.csv');
    A_raw = DD_train( :, 1:10 );
    b_raw = DD_train(:, 11);
    test_index = randSample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% UJIIndoorLoc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% http://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc
% Attribute 521 (Longitude): Longitude. Negative real values from -7695.9387549299299000 to -7299.786516730871000 
% Attribute 522 (Latitude): Latitude. Positive real values from 4864745.7450159714 to 4865017.3646842018. 
if strcmp(dataName, 'UJIIndoorLoc-regression')
    DD_train = xlsread('./data/UJIndoorLoc/trainingData.csv');
    A_train = DD_train( :, 1:520 );
    b_train = DD_train(:, 521);
    DD_test = xlsread('./data/UJIndoorLoc/validationData.csv');
    A_test = DD_test( : , 1:520 );
    b_test = DD_test( : , 521 );
end

% %%%%%%%%% music %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
% Attribute 521 (Longitude): Longitude. Negative real values from -7695.9387549299299000 to -7299.786516730871000 
% Attribute 522 (Latitude): Latitude. Positive real values from 4864745.7450159714 to 4865017.3646842018. 
if strcmp(dataName, 'music')
    allData = importdata('./data/music/YearPredictionMSD.txt');
    A_train = allData(1:463715, 2:end);
    b_train = allData(1:463715, 1);
    A_test = allData( 463716:end, 2:end );
    b_test = allData( 463716:end, 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% END RERGRESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('..................DONE LOADING DATA.................\n');
end

%%%%%%%%%% CT Slice %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DD = importdata('./data/CTSlice/slice_localization_data.csv');
% A = DD.data(:,2:385);
% b = 0.5*(1+sign(randn(size(A,1),1)));
% ubbb = sort(unique(b));
% assert(ubbb(1) == 0 && ubbb(2) == 1);

%%%%%%%%%% Parkinson %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% D = importdata('./data/Parkinson/train_data.txt');
% A = D(:,2:27); b = D(:,29);
% ubbb = sort(unique(b));
% assert(ubbb(1) == 0 && ubbb(2) == 1);

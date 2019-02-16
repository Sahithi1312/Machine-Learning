clc
clear
close 

% each row is a sample
load('PIE.mat');

% number of training data per person
numTrain = 15;

trainInd = [];
testInd = [];

% make train/test index
for i = 1: n_per
        trainInd = [trainInd, (i-1)*n_sub+1: (i-1)*n_sub+numTrain];
        testInd = [testInd, (i-1)*n_sub+numTrain+1: i*n_sub];
end

%generate training and testing data
trainFea = Data(trainInd,:);
trainLabel = Label(trainInd,:);
testFea = Data(testInd,:);
testLabel = Label(testInd,:);
tic;

[rows,columns] = size(trainFea);

% applying PCA first
options=[];
options.ReducedDim=100;
[eigvectorPCA,eigvaluePCA] = PCA(trainFea,options);
trainFea = trainFea * eigvectorPCA;
testFea = testFea * eigvectorPCA;
 
 
% Normalizing Data
trainFea = NormalizeFea(trainFea);
 
%Getting Unique lables out of training label column
uniqueClassLabels = unique(trainLabel); 
 
% for calculating SW scatter within class
inClassScatterSw = 0;
 
%total mean forcalcluating meean of all data for SB calculation
totalDataMean = mean(trainFea);


% for calculating SB  scatter between classes
interClassScatterSb = 0;

% calculating scatter within classes
for i=1:length(uniqueClassLabels)

        %getting index of rows of one class
        indices = find(trainLabel == uniqueClassLabels(i));
        
        % calculating mean of each class
        m = mean(trainFea(indices,:), 1);
        inClassScatterSw = inClassScatterSw+(trainFea(indices,:)-m)' * (trainFea(indices,:)-m);
        ni = length(indices);
        interClassScatterSb = interClassScatterSb + ni * (m - totalDataMean)' * (m - totalDataMean);
end

[eigvector, eigvalue] = eig(interClassScatterSb, inClassScatterSw);
ldaTrainFea = trainFea * eigvector;
ldaTestFea = testFea * eigvector;
ldaTime = toc;

% call nearest neighbor classifier of matlab
predictLabel = knnclassify(ldaTestFea, ldaTrainFea, trainLabel);

acc = sum(predictLabel == testLabel) / length(testLabel);

fprintf('the reconition accuracy with lda is %f.\n', acc);
fprintf('the running time is %f.\n', ldaTime);






tic;

[rows,columns] = size(trainFea);
sigma =(1/rows) * trainFea' * trainFea;
k =100;
[U,S,V] = svd(sigma);
Ureduce = U(:,1:k);   
pcaTime = toc;  
pcaTrainFea = trainFea*Ureduce;
pcaTestFea = testFea * Ureduce;

% call nearest neighbor classifier of matlab
Mdl = fitcknn(pcaTrainFea, trainLabel);
predictLabel = predict(Mdl,pcaTestFea);

acc = sum(predictLabel == testLabel)/length(testLabel);

fprintf('the reconition accuracy with pca is %f.\n', acc);
fprintf('the running time is %f.\n', pcaTime);

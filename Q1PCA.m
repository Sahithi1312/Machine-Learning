clc
clear
close 

% each row is a sample
load('PIE.mat');

% number of training data per person
numTrain = 15;

trainInd = [];
testInd = [];

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
sigma =(1/rows) * trainFea' * trainFea;
k =100;
[U,S,V] = svd(sigma);
Ureduce = U(:,1:k);   
pcaTime = toc;  
pcaTrainFea = trainFea*Ureduce;
pcaTestFea = testFea * Ureduce;

% call nearest neighbor classifier of matlab
predictLabel = knnclassify(pcaTestFea, pcaTrainFea, trainLabel);

acc = sum(predictLabel == testLabel)/length(testLabel);

fprintf('the reconition accuracy with pca is %f.\n', acc);
fprintf('the running time is %f.\n', pcaTime);




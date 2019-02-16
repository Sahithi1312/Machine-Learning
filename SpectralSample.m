clc
clear
close all

% load dataset
load('USPS_sub.mat');
% normalize row feature
data = NormalizeFea(data, 1);

% number of neighbors
num_neighbors = 7;
% number of sample in each block
block_size = 5000;
data = data;
label = label;
[rows,columns] = size(data);
tic;
%% Generate k-nearest-neighbor-graph. 
% An efficient implementation has been provided. 
% You will observe the graph from the workspace
% but knn_dis_graph is not symmetric...

knn_dis_graph = full(gen_dis_graph(data, num_neighbors, block_size));

%% Generate symmetric graph
% You may use A_sym = (A+A')/2
% your code goes here
for i=1 : rows
    [n,d] = knnsearch(data,data(i,:),'k',5);
    for j=1 : length(n)
         knnMatrix(i,n(j)) = d(j) ;
    end
end

degreeMatrix = zeros(rows,columns);

for i=1 : rows
    for j=1 : size(knnMatrix,2)
        if i == j
            degreeMatrix(i,j) = sum(knnMatrix(i,:));
        end
         
    end
end
%% Generate similarity/affinity graph
% consider Gaussian similarity
% your code goes here


%% Generate graph Laplacian
% your code goes here

laplacian = degreeMatrix - knnMatrix;
laplacian = NormalizeFea(laplacian);

%% Eigen-decomposition of graph Laplacian
% your code goes here

[eigvec,eigvalue] = eigs (laplacian);
newData = fliplr(eigvec);
newData = NormalizeFea(newData);
 idx = kmeans(newData,5);
runTime = toc;
% [predictLabel, center] = litekmeans(newData, size(newData,1), 'MaxIter', 100, 'Replicates', 2)
% compute the clustering accuracy
clusteringAcc = accuracy(label, idx);
% compute the clustering NMI
clusteringNMI = nmi(label, idx);
 
fprintf('the clustering accuracy of Kmeans is %f.\n', clusteringAcc/100);
fprintf('the running time of Kmeans is %f seconds.\n', runTime);
fprintf('%f  NMI.\n', clusteringNMI);

%% Pick the first k eigenvectors and treat each row as the new embedding in low dimensional space
% Usually k equals to the number of clusters (ground-truth)
% Use "NomalizeFea()" function to normalize each row
% your code goes here


%% Run Kmeans on the new features (k-dimension feature)
% your code goes here


%% Evaluation
% your code goes here



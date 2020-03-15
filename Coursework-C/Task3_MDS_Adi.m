clear
clc
%read in the data in matlab.
T = readtable('test.csv');
%Compute the dissimilarity matrix.
dissimilarities = squareform(pdist(table2array(T),'squaredeuclidean'));
%Dissimilarity Matrix is a square matrix of dimensions m x m where m is the
%size of the samples.
size(dissimilarities); 
% computing the sample size.
m=size(T,1);
% computing the centering matrix.
centM = eye(m) - (1/m) * ones(m);
% creating the gram matrix through the centering matrix. 
Gram = -.5.*(centM*(dissimilarities)*centM);
%compute eigen values and eigen vectors of the gram matrix.
[eigvec, eigval] =eig(Gram,'matrix');
%computing the indices of the eigen values in decreasing order.
[d,ind] = sort(diag(eigval),'descend');
%sorting eigen values as per the computed indices.
eigvalsorted = eigval(ind,ind);
% similarly sorting the eigen vectors as per the computed indices.
eigvecsorted = eigvec(:,ind);
d=3; % number of dimensions needed by the user.
eigvalneed = eigvalsorted(1:d,1:d); % 2 is the dimension needed.
eigvecneed = eigvecsorted(:,1:d); % 2 is the dimension needed.
X = (sqrt(eigvalneed)*eigvecneed')'; % computing the coordinates of X in reduced dimensionality.
%plot(X(:,1),X(:,2),'o') %plotting the data in reduced dimensionality.
dissimilarities_reduced = squareform(pdist(X,'squaredeuclidean')); %computing the dissimalirity in reduced feature space.
stress = (sqrt(dissimilarities) - sqrt(dissimilarities_reduced))^2; %computing the difference in dissimalirity and squaring it.
overallstress = sum(stress,'all') % computing the final stress value as the sum of squared distances between solution and input distances.


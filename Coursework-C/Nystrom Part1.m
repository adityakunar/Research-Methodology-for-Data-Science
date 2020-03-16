%nystrom method 1.

clear
clc

T = readtable("HighDimDat3.csv");

T = T(1:100,1:10);

data = table2array(T);

data = data - mean(data,1);

Cov = cov(data);

l = 5;

NyCov = Cov(:,1:l);

NyCovA = NyCov(1:l,1:l);


NyCovB = NyCov((l+1):size(Cov,1),1:l);

[eigvecA, eigvalA] =eig(NyCovA,'matrix');
%computing the indices of the eigen values in decreasing order.
[d,ind] = sort(diag(eigvalA),'descend');
%sorting eigen values as per the computed indices.
eigvalsorted = eigvalA(ind,ind);
% similarly sorting the eigen vectors as per the computed indices.
eigvecsorted = eigvecA(:,ind);

U_A = eigvecsorted;

U_B = NyCovB * U_A * inv(eigvalsorted); 

PCA_Mat = [U_A; U_B];

ReducedData = data * PCA_Mat;





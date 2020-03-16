%nystrom method 2.

clear

clc

T = readtable("HighDimDat3.csv");

T = T(1:100,1:10); %

data = table2array(T);

L = [2,4,6,8];

Index = 1 : size(data,2);

FirstL = data(:,L);

RMI = setdiff(Index,L);

RestMat = data(:,RMI);

Newdata = [FirstL RestMat];

Newdata = Newdata - mean(Newdata,1);

Cov = cov(Newdata);

l=size(L,2);

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


PCA = zeros(size(PCA_Mat));

k = 1    
for a = L
    PCA(a,:)=PCA_Mat(k,:);
    k=k+1;
end

for a = RMI
    PCA(a,:)=PCA_Mat(k,:);
    k=k+1;
end

ReducedData = Newdata * PCA;








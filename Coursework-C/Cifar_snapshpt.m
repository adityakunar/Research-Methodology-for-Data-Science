clc
clear

T = readtable("Cifar10_ch3.csv"); %load the data matrix.
X = table2array(T);
X = X(2:end,2:end);
standarized_X= (X - mean(X)); %center the matrix with respect to mean.
Gram = standarized_X * standarized_X';
m = size(standarized_X,1);
[eigvec, eigval] =eig(((Gram)./m),'matrix'); % compute the eigen vectors and eigen values of the covariance matrix.
[d,ind] = sort(diag(eigval),'descend'); % compute the indices of the eigen values in descending order based on diagonal elements of the eigen value matrix.
eigvalsorted = eigval(ind,ind); % sort the eigenvalue matrix using the indices.
eigvecsorted = eigvec(:,ind);  % sort the eigvectors matrix using the indices. This will give us the principle components or modes 
                               %  in descending order of the amount of variance in that particular direction indicated by the modes.
eigvalD = diag(sqrt(inv(eigvalsorted)));
d = 3; %dimension we need to reduce to. 
basisvecs = zeros(size(X,2),d); % initialise empty matrix to store basis vectors of affine spaces in the dimension of the full feature space.
% Extending the eigen vectors calculated from the gram matrix to the full
% feature space dimensionality. 
for i = 1:d
    basis = eigvalD(i).*(standarized_X' * flip(eigvecsorted(:,i),2));
    basisvecs(:,i)=basis/norm(basis);
end
% Orthogonal basis vectors as the dot product is very close to zero.
dot(basisvecs(:,1),basisvecs(:,2))
%Plotting the heatmap. 
heatmap(reshape(basisvecs(:,3),[32,32])')


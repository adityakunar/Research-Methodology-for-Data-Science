clear
clc
Data = table2array(readtable("./Data/HighDimDat3.csv")); % load the data in matlab.
Centered_Data = (Data - mean(Data)); %center the matrix with respect to mean.
size(Centered_Data) % we see that the number of samples m(180) is << than the number of features(670).
Gram = Centered_Data * Centered_Data';
m = size(Centered_Data,1);
[eigvec, eigval] =eig(((Gram)./m),'matrix'); % compute the eigen vectors and eigen values of the covariance matrix.
[d,ind] = sort(diag(eigval),'descend'); % compute the indices of the eigen values in descending order based on diagonal elements of the eigen value matrix.
eigvalsorted = eigval(ind,ind); % sort the eigenvalue matrix using the indices.
eigvecsorted = eigvec(:,ind);  % sort the eigvectors matrix using the indices. This will give us the principle components or modes 
                               %  in descending order of the amount of variance in that particular direction indicated by the modes.

eigvalD = diag(sqrt(inv(eigvalsorted)));

d = 2; %dimension we need to reduce to. 

basisvecs = zeros(size(Data,2),d); % initialise empty matrix to store basis vectors of affine spaces in the dimension of the full feature space.

for i = 1:d
    basis = eigvalD(i).*(Centered_Data' * eigvecsorted(:,i));
    basisvecs(:,i)=basis;
end
% Orthogonal basis vectors as the dot product is very close to zero.
dot(basisvecs(:,1),basisvecs(:,2))

% Computing the data in the new reduced coordinate system based on principle modes.
Reduced_data = Centered_Data * basisvecs;


% The variance as explained by the first 2 PCA modes. We also see that the
% variance captured is very close to 1 meaning the first 2 modes capture
% most of the variance in the data. 
var(Reduced_data)/sum(var(Reduced_data))

% Visualising the data in the reduced space.
figure
axes('LineWidth',0.6,...
    'FontName','Helvetica',...
    'FontSize',8,...
    'XAxisLocation','Origin',...
    'YAxisLocation','Origin')
line(Reduced_data(:,1),Reduced_data(:,2),...
    'LineStyle','None',...
    'Marker','o');
axis equal

% Showing that the new data has mutually uncorrelated columns.
corrcoef(Reduced_data)





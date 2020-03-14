clc
clear
load cities %load the ratings data matrix.
standarized_ratings = (ratings - mean(ratings)); %center the matrix with respect to mean.
Cov= cov(standarized_ratings); %compute the covariance matrix.
m = 329; % number of samples.
cov_mat = (standarized_ratings' * standarized_ratings)./(m); %the natural way to compute it. 
[eigvec, eigval] =eig(Cov,'matrix'); % compute the eigen vectors and eigen values of the covariance matrix.
[d,ind] = sort(diag(eigval),'descend'); % compute the indices of the eigen values in descending order based on diagonal elements of the eigen value matrix.
eigvalsorted = eigval(ind,ind) % sort the eigenvalue matrix using the indices.
eigvecsorted = eigvec(:,ind);  % sort the eigvectors matrix using the indices. This will give us the principle components or modes 
                               %  in descending order of the amount of variance in that particular direction indicated by the modes.
d=2; %choose the dimension you want to reduce to.
eigvalneed = eigvalsorted(1:d,1:d) % 2 dimensions needed. All eigen values are indeed non-negative.

eigvecneed = eigvecsorted(:,1:d) % 2 dimensions needed.
% this shows that eigen vectors computed are unit vectors and orthogonal.
norm(eigvecneed(:,1)) 
norm(eigvecneed(:,2))
dot(eigvecneed(:,1),eigvecneed(:,2))

% Computing the data in the new reduced coordinate system based on principle modes.
Reduced_ratings = ratings * eigvecneed;


% The variance as explained by the first 2 PCA modes. 
var(Reduced_ratings)/sum(var(Reduced_ratings))

% Visualising the data in the reduced space.
figure
axes('LineWidth',0.6,...
    'FontName','Helvetica',...
    'FontSize',8,...
    'XAxisLocation','Origin',...
    'YAxisLocation','Origin')
line(Reduced_ratings(:,1),Reduced_ratings(:,2),...
    'LineStyle','None',...
    'Marker','o');
axis equal

% Showing that the new data has mutually uncorrelated columns.
corrcoef(Reduced_ratings)






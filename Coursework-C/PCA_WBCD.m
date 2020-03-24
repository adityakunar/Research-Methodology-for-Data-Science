clear
clc
%read in the data in matlab.
X = readtable('BC.csv');
T = X( :, 3:end);

labels = X(:, 2);
labels =table2array(labels);
Data = table2array(T);
Centered_Data = (Data - mean(Data))./std(Data); %center the matrix with respect to mean.

Cov= cov(Centered_Data); %compute the covariance matrix.
m = size(Data,1); % number of samples.
cov_mat = (Centered_Data' * Centered_Data)./(m); %the natural way to compute it. 

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
Reduced_data = (Centered_Data*eigvecneed);



% The variance as explained by the first 2 PCA modes. We also see that the
% variance captured is very close to 1 meaning the first 2 modes capture
% most of the variance in the data. 
var(Reduced_data)/sum(var(Reduced_data))


corrcoef(Reduced_data)


plot_2d_scatter(Reduced_data, labels, "PCA of WBCD");

function plot_2d_scatter(x, labels, t)
    classes = unique(labels);
    num_classes = size(classes, 1);
    
    colmap = ["g", "r", "b", "y", "m"];
    
    figure();
    for i = 1:num_classes
        indexes = find(strcmp(labels, classes(i)));
        scatter(x(indexes,1), x(indexes,2),colmap(i));
        hold on
        c = mean(x(indexes,:));
        text(c(1), c(2), classes(i))
        hold on
    
    end
    title(t);
end
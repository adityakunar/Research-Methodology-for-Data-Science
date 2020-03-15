%%
% Nystrom PCA
data = readtable('../Data/iris.data.csv');
X = data( :, 1:4);
labels = data(:, 5);

X = table2array(X);
labels = table2array(labels);

rng('default') % set seed
S = X(randsample(length(X),8),:); %subset S

Y = center_points(S);

C = compute_cov_matrix(Y);


%%
function [x_c] = center_points(x)
    x_c = x - mean(x);
end

%%
function [C] = compute_cov_matrix(X)
    C = X' * X;
end

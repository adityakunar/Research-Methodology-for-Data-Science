data = readtable('./data/iris.data.csv');
X = data( :, 1:4);
labels = data(:, 5);

X = table2array(X);
labels = table2array(labels);

Y = center_points(X);
size(Y,2)

G = compute_gram_matrix(Y);
disp(G)

d=2; % reduced to 2 dimentions
mapping = compute_mapping(G, Y, d);

X_low_dim = X * mapping;


%%

complot = figure();
iris_setosa_indexes = find(strcmp(labels, 'Iris-setosa'));
iris_versicolor_indexes = find(strcmp(labels, 'Iris-versicolor'));
iris_virginica_indexes = find(strcmp(labels, 'Iris-virginica'));

scatter(X_low_dim(iris_setosa_indexes,1), X_low_dim(iris_setosa_indexes,2), 'red')
hold on
scatter(X_low_dim(iris_versicolor_indexes,1), X_low_dim(iris_setosa_indexes,2), 'blue')
hold on
scatter(X_low_dim(iris_virginica_indexes,1), X_low_dim(iris_setosa_indexes,2), 'green')
grid on

%%
function [x_c] = center_points(x)
    x_c = x - mean(x);
end

%%
function [G] = compute_gram_matrix(X)
    G = X * X';
end

%%
function [mapping] = compute_mapping(G, Y, d)
    % G: Gram matrix
    % Y: Centred data
    % d: dimentions to map to
    
    [eigenVec,eigenVal] = eigs(G./size(G,2));
    
    [temp, indexes] = sort(diag(eigenVal), 'descend'); % sort eigenvalue
    eigenVal = eigenVal(:, indexes); % sort eigenvals
    eigenVec = eigenVec(:, indexes); % sort eigenvector
    
    eigenValD = diag(sqrt(inv(eigenVal)));
    
    mapping = zeros(size(Y,2), d);
    for i = 1:d
        basis = eigenValD(i).* (Y' * eigenVec(:,i));
        mapping(:,i) = basis;
    end
end
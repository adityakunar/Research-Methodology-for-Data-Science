data = readtable('./data/iris.data.csv');
X = data( :, 1:4);
labels = data(:, 5);

X = table2array(X);
labels = table2array(labels);

Y = center_points(X);
size(Y,2)

G = compute_gram_matrix(Y);

d=2; % reduced to 2 dimentions
mapping = compute_mapping(G, Y, d);

X_low_dim = X * mapping;

plot_2d_scatter(X_low_dim, labels);


% #######################################################################################################


%%

function plot_2d_scatter(x, labels)
    classes = unique(labels);
    num_classes = size(classes, 1);
    
    colmap = ["r", "g", "b", "y", "m"];
    
    figure();
    for i = 1:num_classes
        indexes = find(strcmp(labels, classes(i)));
    
        scatter(x(indexes,1), x(indexes,2), colmap(i));
        hold on
        c = mean(x(indexes,:));
        text(c(1), c(2), classes(i))
        hold on
    end
end


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
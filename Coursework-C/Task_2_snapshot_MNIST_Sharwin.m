clear all
close all

data = readtable('./data/mnist_train.csv');
X = data( :, 2:(size(data,2)));
labels = data(:, 1);

X = table2array(X);
labels = table2array(labels);

Y = center_points(X);

data_desc = "MNIST ";


%%
disp(unique(labels));
disp(size(X));

%%
% Full blown PCA
plot_labels_distribution(labels, data_desc + "Full dataset target distribution");
[Y, mapping] = Gram_PCA(X, 2);

plot_2d_scatter(Y * mapping, labels,  labels, "PCA with all samples");


% #######################################################################################################
%%

% Test bench (Consistent prior sampling)
sample_rate = 0.1;
[snapshot, snap_labs] = get_constant_prior_snapshot(X, labels, sample_rate); % sample only 20% of datafrom each class
plot_labels_distribution(snap_labs, "Consistent class sampling target distribution");

[snap_Y, snap_mapping] = Gram_PCA(snapshot, 2);

plot_2d_scatter(snap_Y * snap_mapping, snap_labs, labels, join(['Snapshot S-PCA with ', string(sample_rate*100), '% sampling per class'], ""));
plot_2d_scatter(Y * snap_mapping, labels, labels, join(['Full data S-PCA with ', string(sample_rate*100), '% sampling per class'], ""));

%%

% Test bench (Random sampling)
sample_rate = 0.1;
[snapshot, snap_labs] = get_random_snapshot(X, labels, sample_rate); % sample only 20% of datafrom each class
plot_labels_distribution(snap_labs, "Random sampling target distribution");

[snap_Y, snap_mapping] = Gram_PCA(snapshot, 2);

plot_2d_scatter(snap_Y * snap_mapping, snap_labs, labels, join(['Snapshot S-PCA with ', string(sample_rate*100), '% Random sampling'], ""));
plot_2d_scatter(Y * snap_mapping, labels,labels,  join(['Full Data S-PCA with ', string(sample_rate*100), '% Ransom sampling'], ""));


% #######################################################################################################
%%
function [Y, mapping] = Gram_PCA(x, d)
    Y = center_points(x);
    G = compute_gram_matrix(Y);
    mapping = compute_mapping(G, Y, d);
end


%%
function [] = plot_labels_distribution(labs, t)
    width=800;
    height=500;
    
    figure()
    histogram(labs);
    set(gcf,'position',[0,0,width,height]);
    title(t)
end

%% 

function [snapshot, snap_labs] = get_constant_prior_snapshot(x, lab, sampling_ratio)
    classes = unique(lab);
    num_classes = size(classes, 1);
    
    snapshot = [];
    snap_labs = [];
    
    s = RandStream('mlfg6331_64'); 
    
    for i = 1:num_classes
        if isa(lab, 'double') || isa(lab, 'int')
            indexes = find(lab == classes(i)); % type double
        else
            indexes = find(strcmp(lab, classes(i))); % type cell
        end
        k = int16(sampling_ratio * size(indexes,1));
        if k <= 1
            k=2;
        end
        disp(k)
        select_indexes = datasample(s, indexes, k);
        
        snapshot = vertcat(snapshot, x(select_indexes, :));
        snap_labs = horzcat(snap_labs, (repelem(classes(i), size(select_indexes, 1))));
    end
    snap_labs = snap_labs';
end


%% 

function [snapshot, snap_labs] = get_random_snapshot(x, lab, sampling_ratio)
    s = RandStream('mlfg6331_64'); 
    k = int16(sampling_ratio * size(x,1));
    disp(k);
    select_indexes = datasample(s, 1:size(lab,1), k);
    snapshot = x(select_indexes, :);
    snap_labs = lab(select_indexes, :);
end


%%

function plot_2d_scatter(x, lab, orig_lab, t)
    classes = unique(orig_lab);
    num_classes = size(classes, 1);
    
    colmap = lines(num_classes);
    width=1000;
    height=800;
    
    figure();
    c = [];
    for i = 1:num_classes
        if isa(lab, 'double') || isa(lab, 'int')
            indexes = find(lab == classes(i)); % type double
        else
            indexes = find(strcmp(lab, classes(i))); % type cell
        end
        
        if size(indexes,1) <= 0
            continue
        end
        

        scatter(x(indexes,1), x(indexes,2),[], colmap(i,:), 'DisplayName', string(classes(i)), 'MarkerFaceAlpha',.4,'MarkerEdgeAlpha',.4);
        hold on
        c = mean(x(indexes, :));
        text(c(1), c(2), string(classes(i)),'FontSize', 16 )
        hold on
    end
    
    legend()
    set(gcf,'position',[0,0,width,height]);
    title(t);
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
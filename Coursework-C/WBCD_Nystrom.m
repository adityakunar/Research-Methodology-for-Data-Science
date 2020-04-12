clear
clc
%read in the data in matlab.
X = readtable('BC.csv');
T = X(:, 3:end);
labels = X(:,2);
labels =table2array(labels);
data = table2array(T);

s = RandStream('mlfg6331_64'); 

k = int16(.30*size(data,2));

disp(k);

select_indexes = datasample(s, 1:size(data,2), k, 'Replace',false);

L =select_indexes

l=size(L,2);

Index = 1 : size(data,2);

FirstL = data(:,L);

RMI = setdiff(Index,L);

RestMat = data(:,RMI);

Newdata = [FirstL RestMat];

Newdata = Newdata - mean(Newdata,1);

data = data - mean(data,1);

m = size(data,1);

NyCov = (Newdata' * Newdata(:,1:l))./(m-1); %the natural way to compute it. 

NyCovA = NyCov(1:l,1:l);

NyCovB = NyCov((l+1):end,1:l);

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

k = 1;   
for a = L
    PCA(a,:)=PCA_Mat(k,:)/norm(PCA_Mat(k,:));
    k=k+1;
end

for a = RMI
    PCA(a,:)=PCA_Mat(k,:)/norm(PCA_Mat(k,:));
    k=k+1;
end

dot(PCA(:,1),PCA(:,2))

ReducedData = data * PCA;

corrcoef(ReducedData)

var(ReducedData)/sum(var(ReducedData))


plot_2d_scatter(ReducedData, labels, "WBCD Nystrom")

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


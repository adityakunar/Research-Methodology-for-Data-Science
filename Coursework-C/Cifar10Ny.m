clear all

close all

T = readtable("Cifar10_ch3.csv"); %load the data matrix.
X = table2array(T);
X = X(2:end,2:end);

%data = readtable('./data/mnist_train.csv');

%X = data( :, 2:(size(data,2)));

%labels = data(:, 1);



%X = table2array(X);

%labels = table2array(labels);



%Y = center_points(X);



data_desc = "MNIST ";

%%





L = 1:100:784; % selected features



[PCA, Newdata] = calc_mapping(L, X);



disp("dot product = " + dot(PCA(:,1),PCA(:,2)))



ReducedData = Newdata * PCA;



corrcoef(ReducedData)



var(ReducedData)/sum(var(ReducedData));



%plot_2d_scatter(ReducedData, labels, labels, "Nystrom Iris")

%%

%=================================================================================



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

        



        scatter(x(indexes,1), x(indexes,2),[], colmap(i,:), 'DisplayName', string(classes(i)), 'MarkerFaceAlpha',.4,'MarkerEdgeAlpha',.6);

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

function [PCA, Newdata] = calc_mapping(L, data)

    l=size(L,2);





    Index = 1 : size(data,2);



    FirstL = data(:,L);



    RMI = setdiff(Index,L);



    RestMat = data(:,RMI);



    Newdata = [FirstL RestMat];



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

    disp("NyCovB")

    disp(NyCovB(1:10, :))

    

    disp("eigvalsorted")

    disp(eigvalsorted)

    

    disp("inv(eigvalsorted)")

    disp(inv(eigvalsorted))



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

end



%%

function [x_c] = center_points(x)

    x_c = x - mean(x);

end
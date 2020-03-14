% PCA Stuff 

T = readtable('~./Data/HighDimData.csv');

A = table2array(T);

A = A-repmat(mean(A),size(A,1),1)

Cov = cov(A);

[eigvec, eigval] =eig(Cov,'matrix');

[d,ind] = sort(diag(eigval),'descend');

eigvalsorted = eigval(ind,ind)

eigvecsorted = eigvec(:,ind);

eigvalneed = eigvalsorted(1:2,1:2) % 2 is the dimension needed.


eigvecneed = eigvecsorted(:,1:2) % 2 is the dimension needed.


Alow = A * eigvecneed;


plot(Alow(:,1),Alow(:,2),'o');
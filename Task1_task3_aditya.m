
%MDS Stuff

T = readtable('HighDimData.csv');

dissimilarities = squareform(pdist(table2array(HighDimDat)));

size(dissimilarities); 

m=5014;

centM = eye(m) - (1/m) * ones(m);

Gram = -.5*centM*(dissimilarities)*centM;

[eigvec, eigval] =eig(Gram,'matrix')

[d,ind] = sort(diag(eigval),'descend');

eigvalsorted = eigval(ind,ind)

eigvecsorted = eigvec(:,ind);

eigvalneed = eigvalsorted(1:2,1:2) % 2 is the dimension needed.


eigvecneed = eigvecsorted(:,1:2) % 2 is the dimension needed.

X = eigvecneed*sqrt(eigvalneed);

plot(X(:,1),X(:,2),'o')



dissimilarities_reduced = squareform(pdist(table2array(X)));

stress = (dissimilarities - dissimilarities_reduced).*(dissimilarities - dissimilarities_reduced)

overallstress= sum(stress,'all')


% PCA Stuff 


T = readtable('HighDimDat.csv');

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


T = readtable('HighDimDat.csv');

A = table2array(T);

A = A-repmat(mean(A),size(A,1),1);

Gram = (1/5014) * A*A';

[eigvec, eigval] =eig(Gram,'matrix');

[d,ind] = sort(diag(eigval),'descend');

eigvalsorted = eigval(ind,ind);

eigvecsorted = eigvec(:,ind);

eigvalD = diag(sqrt(inv(eigvalsorted)));

basisvec1 = eigvalD(1) * A' * eigvecsorted(:,1:1);
basisvec2 = eigvalD(2) * A' * eigvecsorted(:,2:2);
basisvecs = [basisvec1 basisvec2] ;

Alow = A * basisvecs;



plot(Alow(:,1),Alow(:,2),'o');
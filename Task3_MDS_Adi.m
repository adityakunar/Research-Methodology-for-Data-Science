%MDS Stuff

T = readtable('~./Data/HighDimData.csv');

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

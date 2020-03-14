T = readtable('~./Data/HighDimData.csv');

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
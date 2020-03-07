%Load image and convert it to a vector
image=imread('lena_gray.png');
[h w d]=size(image);
x = double(reshape(image,w*h,d))/255;

%Compute the covariance matrix and its eigenvalues and vectors
C = x*x';
%Computation of all eigenvalues and eigenvectors
%[V,D] = eig(C);
%Computation of the first 4 eigenvalues and eigenvectors
[V,D] = eigs(C,4);

%extract first eigenvector from matrix of eigenvectors
em1=V(:,1);
%project image onto eigenspace
p1x=x'*em1*em1;

%convert eigenvector to image and display the image
image =uint8(reshape(p1x,h,w,d)*255);
figure, imshow(image)


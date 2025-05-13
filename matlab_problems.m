% Load the .mat file
data = load('lp_kb2.mat');

% Access the 'Problem' structure
Problem = data.Problem;

% Extract A and d
A = Problem.A;       % Sparse matrix
d = Problem.aux.c;   % Cost vector

% Create b as a zero vector with the same dimensions as the original b
b = zeros(size(Problem.b)); % Replace b with a zero vector

% Create Q as an identity matrix with dimensions matching d
Q = eye(length(d)); % Identity matrix with the same length as d

% Create F as an identity matrix with dimensions matching the columns of A
F = eye(size(A, 2)); % Identity matrix with size equal to the number of columns in A

% Create c as a vector with the same length as the columns of A
c = zeros(size(A, 2), 1); % Example: initialize c as a zero vector

% Convert A to full matrix if needed
A = full(A);

% Run the IntPoint function
[x, lambda, mu, z, norma, cinter] = IntPoint(Q, A, F, d, b, c);

% Display the outputs
disp('x:'); disp(x);
disp('lambda:'); disp(lambda);
disp('mu:'); disp(mu);
disp('z:'); disp(z);
disp('norma:'); disp(norma);
disp('cinter:'); disp(cinter);

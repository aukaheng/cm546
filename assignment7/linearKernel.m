function sim = linearKernel(x1, x2)
% LINEARKERNEL returns a linear kernel between x1 and x2
%   sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
%   and returns the value in sim


% Compute the linear kernel
sim = x1 * x2';  % dot product

end
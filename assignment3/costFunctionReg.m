function [J, grad] = costFunctionReg(theta, X, y, lambda)
% COSTFUNCTIONREG Compute cost and gradient for logistic regression with
% regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost with respect to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the 
%               partial derivatives of the cost with respect to each
%               parameter in theta.

% Bias theta should not be regularized
theta1 = 0;
theta_reg = [theta1; theta(2:end, :)];

h = sigmoid(X * theta);

regTerm = lambda / (2 * m) * (theta_reg' * theta_reg);

J = (1 / m) * ((-y' * log(h) - (1 - y)' * log(1 - h)) + regTerm);

grad = (1 / m) * ((X' * (h - y)) + (lambda * theta_reg));



% =============================================================

end

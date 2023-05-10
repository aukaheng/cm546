function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
% LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
% regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X 12x2
% theta 2x1
% h 12x1
h = X * theta;

% J 1x1
J = (1 / (2 * m)) * ((h - y)' * (h - y));

% Don't affect the bias term x0
theta0 = 0;

regularizedTheta = [theta0; theta(2:end, :)];

% regularizationTerm 1x1
regularizationTerm = (lambda / (2 * m)) * (regularizedTheta' * regularizedTheta);

J = J + regularizationTerm;

% h-y 12x1
% theta is 2x1, thus its corresponding grad is also 2x1
grad = ((1 / m) * (X' * (h - y))) + ((lambda / m) * regularizedTheta);

% =========================================================================

grad = grad(:);

end

function [theta, J_history] = stochasticGradientDescentMulti(X, y, theta, alpha)
% GRADIENTDESCENTMULTI Performs stochastic gradient descent 
% to learn theta
%   theta = STOCHASTICGRADIENTDESCENTMULTI(x, y, theta, alpha)
%   updates theta by taking one step with learning rate alpha

% Initialize some useful values
m = length(y);  % number of training examples

% ===================== Part 1 - YOUR CODE HERE ===================
% Instructions: Shuffle the training data before use.
%
randomizedRowIndexs = randperm(m);
X_shuffled = X(randomizedRowIndexs, :);
X = X_shuffled;

% =================================================================

J_history = zeros(m, 1);

  for i = 1:m

    % ================== Part 2 - YOUR CODE HERE ==================
    % Instructions: Perform a single gradient step on the parameter
    %               vector theta.
    %
    % Pick one x and y to compute the theta
    xi = X(i, :);
    yi = y(i);

    % The hypothese should be the same as y, which is 1x1
    % X_shuffled is 47x3
    % theta is 3x1
    hypothese = xi * theta;
    error = hypothese - yi;
    gradient = xi' * error;
    %theta = theta - alpha * gradient;
    alpha = implicitUpdate(alpha, xi');
    theta = theta - alpha * gradient;
		
    % =============================================================

    % Save the cost J in every iteration
    J_history(i) = computeCostMulti(X, y, theta);

  end
end

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





% =================================================================

J_history = zeros(m, 1);

  for i = 1:m

    % ================== Part 2 - YOUR CODE HERE ==================
    % Instructions: Perform a single gradient step on the parameter
    %               vector theta.
    % 


 
		
    % =============================================================

    % Save the cost J in every iteration
    J_history(i) = computeCostMulti(X(i, :), y(i), theta);

  end
end

function [error_train, error_val] = ...
    learningCurve(X, y, X_val, y_val, lambda)
% LEARNINGCURVE Generates the train and cross validation set errors needed 
% to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, X_val, y_val, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and error_val(i) should give you the
%               errors obtained after training on i examples.
%
% Note: You should evaluate the training error on the "first i" training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the "entire" cross validation set (X_val and y_val).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0.
%
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using  
%           % training examples X(1:i, :) and y(1:i), then
%           % storing the result in error_train(i) and error_val(i)
%       end
%

% -------------------------- Sample Solution --------------------------

% for i = 1:m
%   X_train = X(1:i, :);
%   y_train = y(1:i);

%   theta = trainLinearReg(X_train, y_train, lambda);

%   pred     = X     * theta;
%   pred_val = X_val * theta;

%   error_train(i) = 1 / (2 * i) * sum((pred(1:i) - y_train) .^2);
%   error_val(i)   = 1 / (2 * size(X_val, 1)) * sum((pred_val - y_val) .^2);
% end

% -------------------------- Your Solution ----------------------------
% Try using linearRegCostFunction for error_train(i) and error_val(i).

for i = 1:m
  X_train = X(1:i, :);
  y_train = y(1:i);

  theta = trainLinearReg(X_train, y_train, lambda);

  pred = X * theta;
  pred_val = X_val * theta;

  error_train(i) = linearRegCostFunction(X_train, y_train, theta, lambda);
  error_val(i) = linearRegCostFunction(X_val, y_val, theta, lambda);
end

% =========================================================================

end

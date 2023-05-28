function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
% COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies * num_features), num_movies, num_features);
Theta = reshape(params(num_movies * num_features + 1:end), ...
                num_users, num_features);
            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it matches
%               our costs. After that, you should implement the gradient
%               and use the checkCostFunction routine to check that the
%               gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies x num_features matrix of movie features
%        Theta - num_users x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Only accumulate cost for user j and movie i only if he/she has rated, which is R(i, j) = 1.
% Theta is 4x3
% X is 5x3
% We need a 5x4 to subtract Y.
% Y is 5x4
% Use R as a filter, we shall be using bitwise operation.
filtered = (X * Theta' - Y) .* R;

% Call sum two times to get a scalar value in octave.
% Required to be a bitwise operation.
J = (1 / 2) * sum(sum(filtered .^ 2));

X_grad = filtered * Theta;

% Theta is 4x3
% filtered is 5x4
% X is 5x3
Theta_grad = filtered' * X;

%
% Regularization
%
costRegularizedTerm = ((lambda / 2) * (sum(sum(Theta .^ 2)))) + ((lambda / 2) * (sum(sum(X .^ 2))));
J = J + costRegularizedTerm;

gradientXRegularizedTerm = lambda * X;
X_grad = X_grad + gradientXRegularizedTerm;

gradientThetaRegularizedTerm = lambda * Theta;
Theta_grad = Theta_grad + gradientThetaRegularizedTerm;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end

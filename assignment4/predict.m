function p = predict(Theta1, Theta2, X)
% PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given
%   the trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, 
%       you can use max(A, [], 2) to obtain the max for each row.
%

% We have two theta, which means we have 3 layer
% The last layer is the h

% bias unit x0
x0 = ones(m, 1);
X1 = [x0, X];

A2 = sigmoid(X1 * Theta1');

% bias unit a(2)0
A2 = [ones(size(A2, 1), 1), A2];

h = sigmoid(A2 * Theta2');

[maxVal, p] = max(h, [], 2);


% =========================================================================


end

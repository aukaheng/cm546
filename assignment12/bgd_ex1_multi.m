%% 
%  Assignment 1: Linear regression with multiple variables
%
%%

clear; close all

fprintf('Loading data ...\n\n');

% Load data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Scale features and set them to zero mean
% Features Normalization
[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Learning Rate
alpha = 0.1;        % Original value = 0.01

% Number of Iterations
num_iters = 50;     % Original value = 400

% Init theta
theta = zeros(3, 1);

fprintf('Running gradient descent ...\n\n');
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

fprintf('Cost (constant alpha): %.2f\n\n', J_history(end))

% Plot the convergence graph
figure;
% numel - returns the number of elements of J_history
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %14.6f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 bedrooms house
price = [1, ([1650, 3] - mu) ./ sigma] * theta;

fprintf(['Predicted price: house 1650 ft2, 3 rooms ' ...
         '(using gradient descent):\n $%.2f\n'], price);

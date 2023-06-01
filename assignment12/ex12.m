%%
%  Assignment 12: Stochastic Gradient Descent (multiple variables)
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the linear
%  assignment. You will need to complete the following functions in
%  this assignment:
%
%     stochasticGradientDescentMulti.m
%     implicitUpdate.m
%
%%

% Running the assignment 1 Batch Gradient Descent
fprintf('\n');
fprintf('Running the assignment 1 with Batch Gradient Descent.\n\n');
bgd_ex1_multi

fprintf('\n');
fprintf('Finish Assignment 1 by running Batch Gradient Descent.\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

% Clear all data and start with a clean working environment
fprintf('\n\n');
fprintf('Clearing data ...\n\n');
clear

% Load data
fprintf('Reloading data ...\n\n');
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
alpha = 0.095;        % Original value = 0.01

% Init theta
theta = zeros(3, 1);

fprintf('Running implicit stochastic gradient descent ...\n\n');
[theta, J_history] = stochasticGradientDescentMulti(X, y, theta, alpha);

fprintf('\nCost (implicit alpha): %.2f\n\n', J_history(end))

% Plot the convergence graph
figure;
% numel - returns the number of elements of J_history
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display implicit stochastic gradient descent's result
fprintf('Theta computed from stochastic gradient descent: \n');
fprintf(' %14.6f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 bedrooms house
price = [1, ([1650, 3] - mu) ./ sigma] * theta;

fprintf(['Predicted price: house 1650 ft2, 3 rooms ' ...
         '(using implicit stochastic gradient descent):\n $%.2f\n'], price);

fprintf('\nProgram paused. Press enter to finish.\n\n');
pause;
close all;
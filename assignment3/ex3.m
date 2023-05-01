%% 
%  Assignment 3: Logistic Regression with Regularization
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the assignment
%  which covers regularization with logistic regression.
%
%  You will need to complete the following function in this assignment:
%
%     costFunctionReg.m
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex3data.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

plotData(X, y);

% Put some labels 
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1 (accepted)', 'y = 0 (rejected)')
hold off;


%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic 
%  regression to classify the data points. 
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features
fprintf('Number of Features, including the Bias (Intercept) Term\n');
% "+1" to include X0 in the counting
fprintf('  Before Polynomial Expansion : %2d\n', size(X, 2) + 1);
% mapFeature will add the intercept term for you
X = mapFeature(X(:,1), X(:,2));
fprintf('  After  Polynomial Expansion : %2d\n', size(X, 2));

fprintf('\nDisplay a slice of X with 10 rows and 5 cols\n')
disp(X([1:10], [1:5]))
fprintf('\n')

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized
% logistic regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n\n');
pause;

%% ============= Part 2: Regularization and Accuracies =============
%  In this part, you will get to try different values of lambda and 
%  see how regularization affects the decision boundary.
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda?
%  How does the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;  % Try 0, 1, 10, 100

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t) costFunctionReg(t, X, y, lambda), initial_theta, options);
	
% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1 (accepted)', 'y = 0 (rejected)', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n\n', mean(p == y) * 100);

fprintf('Program paused. Press enter to continue.\n\n');
pause;

% Make predictions

fprintf('Making prediction for the following Example:\n');
fprintf('         X1        X2\n');
fprintf('   -0.13882  -0.27266\n');

% ====================== YOUR CODE HERE ======================
%
%                      Reference
%                      Solution
%        X1        X2     y
%  -0.13882  -0.27266     1

X1 = 0;     % You should implement X1
X2 = 0;     % You should implement X2
X_map = 0;  % You should implement X_map

fprintf('\nPredict\nY = ');
disp(predict(theta, X_map))

% ============================================================

fprintf('\n');

fprintf('Making prediction for the following Examples:\n');
fprintf('         X1        X2\n');
fprintf('   -0.21947, -0.01681\n');
fprintf('   -0.13882, -0.27266\n');
fprintf('    0.18376,  0.93348\n');

% ====================== YOUR CODE HERE ======================
%
%                      Reference
%                      Solution
%        X1        X2     y
%  -0.21947  -0.01681     1
%  -0.13882  -0.27266     1
%   0.18376   0.93348     0

X1 = 0;     % You should implement X1
X2 = 0;     % You should implement X2
X_map = 0;  % You should implement X_map

fprintf('\nPredict\nY =\n');
disp(predict(theta, X_map))

% ============================================================

fprintf('\nProgram paused. Press enter to finish.\n\n');
pause;
close all;
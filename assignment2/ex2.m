%% 
%  Assignment 2: Logistic Regression
%  
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression assignment. You will need to complete the following
%  functions in this assignment:
%
%     plotData.m
%     sigmoid.m
%     costFunction.m
%     predict.m
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('ex2data.txt');
X = data(:, [1, 2]);
y = data(:, 3);

%% ==================== Part 1: Plotting ====================
%  We start the assignment by first plotting the data to understand 
%  the problem we are working with. You need to complete the code
%  in plotData.m

fprintf(['Plotting data with\n' ...
         '  + indicating (y = 1) examples\n' ...
         '  o indicating (y = 0) examples\n']);

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n\n');
pause;

%% ============ Part 2: Compute Cost and Gradient ============
%  In this part of the assignment you will implement the cost and gradient
%  for logistic regression. You need to complete the code in costFunction.m

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add bias term to X
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n\n');
pause;

%% ============= Part 3: Optimizing using fminunc  =============
%  In this part of the assignment you will use "fminunc", an Octave
%  built-in function, to find the optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t) costFunction(t, X, y), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n\n');
pause;

%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use them to predict
%  the outcomes on unseen data. In this part, you will use the logistic
%  regression model to predict the probability that a student with 
%  score 45 on exam 1 and score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies 
%  of our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1 and 
%  score 85 on exam 2
prob = sigmoid([1 45 85] * theta);
fprintf('For a student with scores 45 and 85,\n');
fprintf('we predict an admission probability of %.2f%%\n', prob * 100);

fprintf('Will the student be admitted? %d', predict(theta, [1 45 85]));
fprintf('  (1 = Yes;  0 = No)\n\n')

% Compute accuracy on our training set
p = predict(theta, X);
fprintf('\nTrain Accuracy: %f\n', mean(double(p == y)) * 100);

fprintf('\nProgram paused. Press enter to finish.\n\n');
pause;
close all;

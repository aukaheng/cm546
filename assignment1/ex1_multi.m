%% 
%  Assignment 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the linear
%  assignment. You will need to complete the following functions in
%  this assignment:
%
%     featureNormalize.m
%     computeCostMulti.m
%     gradientDescentMulti.m
%     normalEqn.m
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%d %d], y = %d \n', [X(1:10,:) y(1:10,:)]');

fprintf('\nProgram paused. Press enter to continue.\n\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

%% ================ Part 2: Gradient Descent ================

% ====================== TRY YOURSELF =======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 bedrooms house.
%
% Note 1: By using the 'hold on' command, you can plot multiple graphs
%       on the same figure.
%
% Note 2: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n\n');

% Try different alpha values
% e.g. 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 1.1, 1.3, 2
alpha = 0.1;         % Original 0.01 ---- choose 0.1

% Try different number of iterations
% e.g. 400, 100, 50, 20, 10
num_iters = 50;   % Original 400  ---- choose 50

% Init theta and run gradient descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
% numel - returns the number of elements of J_history
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% More plots - Uncomment the following lines
% [theta, J1] = gradientDescentMulti(X, y, theta, 0.01, num_iters);
% [theta, J2] = gradientDescentMulti(X, y, theta, 0.03, num_iters);
% [theta, J3] = gradientDescentMulti(X, y, theta, 0.1 , num_iters);
% fprintf('\nProgram paused. Press enter to continue.\n\n');
% pause;
% plot(1:50, J1(1:50), 'b');
% hold on;
% plot(1:50, J2(1:50), 'r');
% plot(1:50, J3(1:50), 'k');
% hold off;

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %14.6f', theta);
fprintf('\n\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the
%               gradient descent. You should complete the code 
%               in computeCostMulti.m and gradientDescentMulti.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 bedrooms
%               house.
%
%               Recall: the first column of X is all-ones. Thus,
%                       it does not need to be normalized.

price = 0;  % You need to implement the price
x = [1650, 3];
norm_x = (x - mu) ./ sigma;

%fprintf('Size of norm_x:\n');
%disp(size(norm_x));
%fprintf('Size of theta:\n');
%disp(size(theta));

price = [1, norm_x] * theta;


% ======================================================

fprintf(['Predicted price: house 1650 ft2, 3 rooms ' ...
         '(using gradient descent):\n $%.2f'], price);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations =================

fprintf('Solving with normal equations...\n\n');

%% Load data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %14.6f', theta);
fprintf('\n\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 bedrooms house.
%


price =  0;  % You need to implement the price

%fprintf('Size of theta:\n');
%disp(size(theta));
x = [1650, 3];
price = [1, x] * theta;

% ============================================================

fprintf(['Predicted price: house 1650 ft2, 3 rooms ' ...
         '(using normal equations):\n $%.2f\n'], price);

fprintf('\nProgram paused. Press enter to finish.\n\n');
pause;
close all;
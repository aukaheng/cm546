%% 
%  Assignment 6: Regularized Linear Regression and Bias-Variance
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the assignment.
%  You will need to complete the following functions:
%
%     linearRegCostFunction.m
%     learningCurve.m
%     validationCurve.m
%

%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and 
%  plot the data.
%

% Load Train Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex6data.mat');

% m = Number of examples
m = size(X, 1);

% Plot train data
figure(1)
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 2: Regularized Linear Regression Cost =============
%  You should now implement the cost function for regularized linear 
%  regression. 
%

theta = [1; 1];  lambda = 1;
J = linearRegCostFunction([ones(m, 1) X], y, theta, lambda);

fprintf(['Cost at theta = [1; 1]: %f '...
         '\n(this value should be about 303.993192)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========= Part 3: Regularized Linear Regression Gradient ==========
%  You should now implement the gradient for regularized linear 
%  regression.
%

theta = [1; 1];  lambda = 1;
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, lambda);

fprintf(['Gradient at theta = [1; 1]: [%f; %f] '...
         '\n(this value should be about [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 4: Train Linear Regression =============
%  Once you have implemented the cost and gradient correctly, the
%  trainLinearReg function will use your cost function to train 
%  regularized linear regression.
% 
%  Write Up Note: The data is non-linear, so this will not give a  
%                 great fit.
%

%  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X] * theta, '--', 'LineWidth', 2)
hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 5: Learning Curve for Linear Regression =============
%  Next, you should implement the learningCurve function. 
%
%  Write Up Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" 
%

lambda = 0;
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

figure(2)
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of train examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Train Examples\tTrain Error\tCV Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
close all;

%% =========== Part 6: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. You should now
%  complete polyFeatures to map each example into its powers
%

p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = (X_poly_test - mu) ./ sigma;
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];  % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = (X_poly_val - mu) ./ sigma;
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];  % Add Ones

fprintf('Normalized Train Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with 
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%

lambda = 0; % try values between 0 and 100 (1 is the best fit)
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot train data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of train examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Train Examples\tTrain Error\tCV Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ======== Part 8: Cross Validation for Selecting Lambda =========
%  You will now implement validationCurve to test various values of 
%  lambda on a cv set. You will then use this to select the
%  "best" lambda value.
%

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

figure(3)
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
title(sprintf('Polynomial fit, (lambda = %f)', lambda));

fprintf('lambda\t\tTrain Error\tCV Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Computing Test Set Error =======================
%  To get a better indication of the model's performance in the real world,
%  it is important to evaluate the "final" model on a test set that was not
%  used in any part of training (that is, it was neither used to select the
%  lambda parameters, nor to learn the model parameters theta). You should
%  compute the test error using the best value of lambda you found. 
%

min = exp(30); % accepted test error threshold value
index = 1;     % initialize the index
for i = 1:length(lambda_vec)
    if(min > abs(error_train(i) - error_val(i)))
        min = abs(error_train(i) - error_val(i));
        index = i; % save the index of the minimum of |train - val|
    end
end

lambda = lambda_vec(index);
fprintf('Selected Lambda: %f\n', lambda);

theta = trainLinearReg(X_poly, y, lambda);

lambda = 0;  % Deactivate regularization for testing purpose
error_test = linearRegCostFunction(X_poly_test, ytest, theta, lambda); 
fprintf('Test Error: %f\n', error_test);
fprintf('Program paused. Press enter to continue.\n');
pause;
		
%% ===== Plotting Learning Curves with Randomly Selected Examples =====
%  To determine the training error and cross validation error for i 
%  examples, you should first randomly select i examples from the 
%  training set and i examples from the cross validation set. You 
%  will then learn the parameters theta using the randomly chosen
%  training set and evaluate the parameters θ on the randomly chosen
%  training set and cross validation set. The above steps should then
%  be repeated multiple times (say 50) and the averaged error should
%  be used to determine the training error and cross validation error
%  for i examples.
%

lambda = 0.01;
m = size(X, 1);
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
steps = 50;
draw_pause = 1;  % pause at the end of each step
for i = 1:steps
	fprintf('\nStep: %d\n', i)
    for j = 1:m
        seq = randperm(m, j); % draw j random values from [1,m]
        X_poly_rand = X_poly(seq, :);
        y_rand = y(seq, :);
				
        seq_val = randperm(m, j);
        X_poly_val_rand = X_poly_val(seq_val, :);
        yval_rand = yval(seq_val, :);
        
        theta = trainLinearReg(X_poly_rand, y_rand, lambda);
				
        no_lambda = 0;
        J = linearRegCostFunction(X_poly_rand, ...
                                       y_rand, ...
                                        theta, ...
                                  no_lambda);
        Jval = linearRegCostFunction(X_poly_val_rand, ...
                                           yval_rand, ...
                                               theta, ...
                                     no_lambda);
        error_train(j) = error_train(j) + J;
        error_val(j)   = error_val(j)   + Jval;
    end

	plot(1:m, error_train / steps, 1:m, error_val / steps);
	title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
	legend('Train', 'Cross Validation')
	xlabel('Number of train examples')
	ylabel('Error')
	axis([0 13 0 100])
	drawnow

	if draw_pause != 27  % 27 = ESC key
		fprintf('\nPress ESC to stop pausing at each step.\n')
		draw_pause = kbhit;
	end
end

fprintf('\nProgram paused. Press enter to finish.\n\n');
pause;
close all;
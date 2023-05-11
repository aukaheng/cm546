%% 
%  Assignment 7: Support Vector Machines
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the assignment.
%  You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%

%% Initialization
clear ; close all; clc

%% =============== Part 1: Loading and Visualizing Data ================
%  We start the assignment by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex7data1: 
% You will have X, y in your environment
load('ex7data1.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Training Linear SVM ====================
%  The following code will train a linear SVM on the dataset and plot the
%  decision boundary learned.
%

% Load from ex7data1: 
% You will have X, y in your environment
load('ex7data1.mat');

fprintf('\nTraining Linear SVM ...\n')

% tol is a tolerance value used for determining equality of 
% floating point numbers.
tol = 1e-3;

% You should try to change the C value below and see how the decision
% boundary varies (e.g., try C = 100)
C = 1;
model = svmTrain(X, y, C, @linearKernel, tol, 20);
visualizeBoundaryLinear(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 3: Implementing Gaussian Kernel ===============
%  You will now implement the Gaussian kernel to use with
%  the SVM. You should complete the code in gaussianKernel.m
%
fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1=[1 2 1], x2=[0 4 -1]\nsigma=2: ' ...
         '%f\n(this value should be about 0.324652)\n'], sim);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 4: Visualizing Dataset 2 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex7data2: 
% You will have X, y in your environment
load('ex7data2.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
%  After you have implemented the kernel, we can now use it to train the 
%  SVM classifier.
% 
fprintf('\nTraining SVM with RBF Kernel (may take 1-2 minutes)\n');
fprintf('Note: RBF = Radial Basis Function\n')

% Load from ex7data2: 
% You will have X, y in your environment
load('ex7data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will
% run faster. However, in practice, you will want to run the training 
% to convergence.
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 6: Visualizing Dataset 3 ================
%  The following code will load the next dataset into your environment 
%  and plot the data. 
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex7data3: 
% You will have X, y in your environment
load('ex7data3.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
%  This is a different dataset that you can use to experiment with. Try
%  different values of C and sigma here.
% 

% Load from ex7data3: 
% You will have X, y in your environment
load('ex7data3.mat');

% Trying different SVM Parameters
[C, sigma] = dataset3Params(X, y, Xval, yval);

fprintf('Optimal C = %.2f and Sigma = %.2f\n', C, sigma);

% Train the SVM
fprintf('\n\nSVM');
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('\nProgram paused. Press enter to finish.\n\n');
pause;
close all;
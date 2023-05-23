%% 
%  Assignment 10: Anomaly Detection
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the assignment.
%  You will need to complete the following functions:
%
%     estimateGaussian.m
%     multivariateGaussian.m
%     selectThreshold.m
%     visualizeFit.m
%

%% Initialization
clear ; close all; clc

%% ================== Part 1: Load Example Dataset  ===================
%  We start this assignment by using a small dataset that is easy to
%  visualize.
%
%  Our example case consists of 2 network server statistics across
%  several machines: the latency and throughput of each machine.
%  This assignment will help us find possibly faulty machines.
%

fprintf('Visualizing example dataset for outliers detection.\n\n');

%  The following command loads the dataset. You should now have the
%  variables X, Xval, yval in your environment
load('ex10data1.mat');

%  Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause


%% =============== Part 2: Estimate the dataset statistics ===================
%  For this assignment, we assume a Gaussian distribution for the dataset.
%
%  We first estimate the parameters of our assumed Gaussian distribution, 
%  then compute the probabilities for each of the points and then visualize 
%  both the overall distribution and where each of the points falls in 
%  terms of that distribution.
%
fprintf('Visualizing Gaussian fit.\n\n');

%  Estimate my and sigma2
[mu sigma2] = estimateGaussian(X);

%  Returns the density of the multivariate normal at each data point (row) 
%  of X
p = multivariateGaussian(X, mu, sigma2);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Part 3: Find Outliers ===================
%  Now you will find a good epsilon threshold using a cross-validation set
%  probabilities given the estimated Gaussian distribution
% 

pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon F1] = selectThreshold(yval, pval);

fprintf('\n1st Dataset\n')
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('  (you should see a value epsilon of about 8.990853e-05)\n');
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('# Outliers found: %d\n\n', sum(p < epsilon));

%  Find the outliers in the training set and plot the
outliers = find(p < epsilon);

%  Draw a red circle around those outliers
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Part 4: Multidimensional Outliers ===================
%  We will now use the code from the previous part and apply it to a 
%  harder problem in which more features describe each datapoint and only 
%  some features indicate whether a point is an outlier.
%

%  Loads the second dataset. You should now have the
%  variables X, Xval, yval in your environment
load('ex10data2.mat');

%  Apply the same steps to the larger dataset
[mu sigma2] = estimateGaussian(X);

%  Training set 
p = multivariateGaussian(X, mu, sigma2);

%  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2);

%  Find the best threshold
[epsilon F1] = selectThreshold(yval, pval);

fprintf('\n2nd Dataset\n')
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('  (you should see a value epsilon of about 1.377229e-18)\n');
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('# Outliers found: %d\n', sum(p < epsilon));

fprintf('\nProgram paused. Press enter to finish.\n\n');
pause;
close all;
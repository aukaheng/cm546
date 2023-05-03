%% 
%  Assignment 4: Neural Networks
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the assignment.
%  You will need to complete the following functions in this assignment:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%

%% Initialization
clear; close all; clc

%% Setup parameters for this part
input_layer_size  = 400;  % 20 x 20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the assignment by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex4data.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Loading Parameters ================
% In this part of the assignment, we load some pre-initialized 
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

%% ================= Part 3: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  To give you an idea of the network's output, you can also run
%  through the examples one at a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

do
	% Select randomly an example
    i = randi([1, m]);
	
    % Display an image
    fprintf('\nDisplaying Example Image\n');
    displayData(X(i, :));

    % Predict the number
    pred = predict(Theta1, Theta2, X(i,:));
    fprintf( '\nNeural Network Prediction: %d (real digit %d)\n', ...
		      mod(pred, 10), mod(y(i), 10) );
    
    % Press ESC (27) to finish.
    fprintf('Press ESC to finish.\n');
	keyhit = kbhit();
until(keyhit == 27)

close all;

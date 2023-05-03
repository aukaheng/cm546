function ocrFileImg(f)
% OCRFILEIMG Reads an image containing handwritten number and
%      displays it on screen, then use Neural Network with
%      pre-trained Weights to predict the number.
%
%    Syntax:  ocrFileImg(<filename.jpg>)
%    Example: ocrFileImg('20x20_3.jpg')

% Environment clean up
close all; clc

% Load JPG image
image = imread(f);

% Display the image in Grayscale
colormap(gray)
imagesc(image)
axis image off

% Convert integer to double
Example = double(image(:));

% disp(size(Example));

% Load Weights (Theta1 & Theta2) from MAT file
load('ex4weights.mat');


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network.


% Predict the number
% Example is 400x1
pred = predict(Theta1, Theta2, Example');


% =========================================================================

fprintf('\nNeural Network Prediction: %d\n', mod(pred, 10));
				 
end
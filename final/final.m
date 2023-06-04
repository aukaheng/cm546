clear ; close all; clc

% X is 15000 x 4096
% y is 15000 x 1
load('mnist.mat');

% Shuffle
m = size(X, 1);
randomizedRowIndexs = randperm(m);

X_shuffled = X(randomizedRowIndexs, :);
y_shuffled = y(randomizedRowIndexs, :);

X_train = X_shuffled(1:14950, :);
X_test = X_shuffled(14951:end, :);
X = X_train;
y_train = y_shuffled(1:14950, :);
y_test = y_shuffled(14951:end, :);
y = y_train;

% Define the number of input, hidden, and output layers
inputLayerSize = 4096;
hiddenLayerSize = 50;
outputLayerSize = 15;

initialTheta1 = randInitializeWeights(inputLayerSize, hiddenLayerSize);
initialTheta2 = randInitializeWeights(hiddenLayerSize, outputLayerSize);

% Initialize the weights and bias
initialParameters = [initialTheta1(:); initialTheta2(:)];

% Choosing sigmoid as the activation function

% Forward propagation
% Compute the error
% Backward propagation
% Repeat the above 3 steps 400 times until the error converges
options = optimset('MaxIter', 100);

% Regularization
lambda = 0.25;

costFunction = @(p) nnCostFunction(p, inputLayerSize, hiddenLayerSize, outputLayerSize, X, y, lambda);

[parameters, cost] = fmincg(costFunction, initialParameters, options);

%save('-mat', 'parameters.mat', 'parameters');

Theta1 = reshape(parameters(1:hiddenLayerSize * (inputLayerSize + 1)), hiddenLayerSize, (inputLayerSize + 1));

Theta2 = reshape(parameters((1 + (hiddenLayerSize * (inputLayerSize + 1))):end), outputLayerSize, (hiddenLayerSize + 1));

prediction = predict(Theta1, Theta2, X_test);

result = double(prediction == y_test);

fprintf('\nResult\n');
disp([y_test, prediction, result]);

fprintf('\nAccuracy: %f\n', mean(result) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

close all;
clear; close all; clc

% X is 15000 x 4096
% y is 15000 x 1
load('final.mat');

% Shuffle
examples = size(X, 1);
randomizedRowIndexs = randperm(examples);

X_shuffled = X(randomizedRowIndexs, :);
y_shuffled = y(randomizedRowIndexs, :);

test = 20;
m = examples - 20;

X_train = X_shuffled(1:m, :);
X_test = X_shuffled(m+1:end, :);
X = X_train;
y_train = y_shuffled(1:m, :);
y_test = y_shuffled(m+1:end, :);
y = y_train;

% Define the number of input, hidden, and output layers
inputLayerSize = 4096;
hiddenLayerSize = 50;
outputLayerSize = 11; % 0 - 10

initialTheta1 = randInitializeWeights(inputLayerSize, hiddenLayerSize);
initialTheta2 = randInitializeWeights(hiddenLayerSize, outputLayerSize);

% Initialize the weights and bias
initialParameters = [initialTheta1(:); initialTheta2(:)];

% Choosing sigmoid as the activation function

% Forward propagation
% Compute the error
% Backward propagation
% Repeat the above 3 steps 400 times until the error converges
options = optimset('MaxIter', 50);

% Regularization
lambda = 1;

costFunction = @(p) nnCostFunction(p, inputLayerSize, hiddenLayerSize, outputLayerSize, X, y, lambda);

[parameters, J_history] = fmincg(costFunction, initialParameters, options);

%save('-mat', 'parameters.mat', 'parameters');

plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

Theta1 = reshape(parameters(1:hiddenLayerSize * (inputLayerSize + 1)), hiddenLayerSize, (inputLayerSize + 1));

Theta2 = reshape(parameters((1 + (hiddenLayerSize * (inputLayerSize + 1))):end), outputLayerSize, (hiddenLayerSize + 1));

prediction = predict(Theta1, Theta2, X_train);

result = double(prediction == y_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(result) * 100);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

fprintf('\nNow let''s test the Testing Set.\n');
pause;

result = predict(Theta1, Theta2, X_test);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(result == y_test)) * 100);
fprintf('Result\n');
disp([int32(y_test), int32(result), double(y_test == result)]);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

close all;
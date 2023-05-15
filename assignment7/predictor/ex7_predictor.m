%% Initialization
clear; close all; clc

% Define output values in a cell (i.e. values inside curly braces)
class = {'Ham','Spam'};

fprintf('Loading model. Please wait...\n\n')

% Load saved Model
load('model.mat')
whos

% List the structure of the Model
fprintf('Structure of the Model: modelLinearKernel\n')
disp(fieldnames(modelLinearKernel))
fprintf('\n')

do
	filename = input('Input filename: ', 's');
	file_content = readFile(filename);
	
	if !isempty(file_content)

% ====================== YOUR CODE HERE =======================

		% Prediction
		word_indices = processEmail(file_content);
		X            = emailFeatures(word_indices);
		prediction   = svmPredict(modelLinearKernel, X);

% =============================================================
	
		% Output the processed filename
		fprintf('\nProcessed: %s', filename);
        
        % Output Prediction
        % Note: "1 + prediction" because Octave's index 
        %       starts from 1 and not zero
		fprintf('\n\nTest Result: %s\n\n', class{1 + prediction});
	end
	
until(isempty(file_content))



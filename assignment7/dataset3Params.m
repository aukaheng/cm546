function [C, sigma] = dataset3Params(X, y, Xval, yval)
% EX6PARAMS returns your choice of C and sigma for Part 3 of the assignment
% where you select the optimal (C, sigma) learning parameters to use for SVM
% with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C
%   and sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables (C and sigma) correctly.
C     = 1;
sigma = 0.3;

C_opt     = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_opt = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
error_cv  = zeros(size(C_opt, 2), size(sigma_opt, 2));

minimumError = 1;
optimalC = 1;
optimalSigma = 1;

for i = 1:size(C_opt, 2)
  for j = 1:size(sigma_opt, 2)

    C     = C_opt(1, i);
    sigma = sigma_opt(1, j);

    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    
    % predictions is a vector containing all
    % the predictions from the SVM
    predictions = svmPredict(model, Xval);

    % Error evaluation on the cross validation set
    error_cv(i, j) = mean(double(predictions ~= yval));
		
    %% Uncomment the 3 lines with fprintf to display:
    %% C, sigma, and cross validation error
    fprintf('    (i:%d) C      = %5.2f\n', i, C);
    fprintf('    (j:%d) Sigma  = %5.2f', j, sigma);
    fprintf('    CV Err = %.3f\n\n', error_cv(i, j));

    if (error_cv < minimumError)
      optimalC = C;
      optimalSigma = sigma;
      minimumError = error_cv;
    endif

  end
end

% ====================== YOUR CODE HERE ======================
% Instructions: Analize the code above and return the optimal C
%               and sigma learning parameters found using the 
%               cross validation set.
%

C = optimalC;
sigma = optimalSigma;

% =========================================================================

end

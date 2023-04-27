function plotDecisionBoundary(theta, X, y)
% PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure
% with the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   either:
%   1) m x 3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) m x n, n > 3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:, 2:3), y);

hold on

% Display the number of features
% size(X, 2) <= 3
	
% Only need 2 points to define a line, so choose two endpoints
% plot_x = [28.059, 101.828]
plot_x = [min(X(:, 2)) - 2,  max(X(:, 2)) + 2];

% Calculate the decision boundary line (Newton's Method)
% plot_y = [96.166, 20.653]
plot_y = (-1 ./ theta(3)) .* (theta(2) .* plot_x + theta(1));

% Plot, and adjust axes for better viewing
plot(plot_x, plot_y)		
    
% Legend, specific for the exercise
legend('Admitted', 'Not admitted', 'Decision Boundary')
axis([30, 100, 30, 100])

hold off

end

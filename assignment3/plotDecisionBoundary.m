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
% size(X, 2) > 3
		
% Here is the grid range
u = linspace(-1, 1.5, 50);
v = linspace(-1, 1.5, 50);

z = zeros(length(u), length(v));
% Evaluate z = theta * x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        z(i,j) = mapFeature(u(i), v(j)) * theta;
    end
end
z = z'; % important to transpose z before calling contour

% Plot z = 0
% Notice you need to specify the range [0, 0]
% [0, 0] means no contour spacing is defined, nor the
%        number of contour lines; the only line drawn is
%        the line defined by u, v, and z
contour(u, v, z, [0, 0], 'LineWidth', 2)

hold off

end

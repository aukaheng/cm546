function idx = findClosestCentroids(X, centroids)
%vFINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X, 1), 1);

% Initializing the variable m for both snippets.
m = size(X, 1);

% This snippet runs faster.
distortion = zeros(m, K);
for k = 1:K
    centroidsRepMat = repmat(centroids(k, :), m, 1);
    d = X - centroidsRepMat;
		
    % Calculate the distance from the Example to each Centroid.		
    distortion(:, k) = sum(d.^2, 2);
end
[_, idx] = min(distortion, [], 2);


end


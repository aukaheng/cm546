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

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X, 1);

% For every example
for i = 1:m
  xi = X(i, :);
  distances = zeros(K, 1);

  % Compare it with every centroid
  for j = 1:K
    muj = centroids(j, :);
    distances(j) = norm(xi - muj) ^ 2;
  end

  % https://docs.octave.org/v4.0.3/Utility-Functions.html#XREFmin
  [minimumValue, ci] = min(distances);

  idx(i) = ci;
end

% =============================================================

end


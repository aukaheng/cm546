function p = multivariateGaussian(X, mu, Sigma2)
% MULTIVARIATEGAUSSIAN Computes the probability density function of the
% multivariate gaussian distribution.
%    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
%    density function of the examples X under the multivariate gaussian 
%    distribution with parameters mu and Sigma2. If Sigma2 is a matrix,
%    it is treated as the covariance matrix. If Sigma2 is a vector, it 
%    is treated as the sigma^2 values of the variances in each dimension
%    (a diagonal covariance matrix).
%

n = length(mu);

if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
    Sigma2 = diag(Sigma2);
end

diff = X - mu(:)';

p = (2 * pi) ^ (- n / 2) * det(Sigma2) ^ (-0.5) * ...
    exp(-0.5 * sum(diff * pinv(Sigma2) .* diff, 2));

end
function alpha = implicitUpdate(alpha, X)
% IMPLICITUPDATE implements a Closed-Form Special Case of
% an implicit update of the Learning Rate. This process will
% remain numerically stable virtually for all alpha as the
% learning rate is now normalized.
%   IMPLICITUPDATE(ALPHA) returns a more stable learning rate.
%
%   Robustness - Empirically, implicit updates outperform
%   or nearly match the performance of explicit updates in
%   general.  Furthermore, the implicit methods appear to be
%   more robust to scaling of the data. Using stochastic
%   gradient, the update naturally factors in the scale of
%   the input vector X when computing the update, which may
%   help explain such robustness.
%

% ====================== YOUR CODE HERE ====================
% Instructions: Implement the Closed-Form Special Case of
%               an implicit update of the Learning Rate
%               following the formula described in the
%               p.3 of the paper "Implicit Online Learning",
%               K. Brian and B. Peter

% η is the learning rate
% The X is actually X(i, :)
alpha = alpha / (1 + alpha * X .^ 2);



% ========================================================== 
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% original X is [12x1]; y is[12x1]; theta is [2x1]
% the input X to this function is [ones(m, 1) X], which is [12x2]
h = X * theta; % [12x2]*[2x1] = [12x1]
reg = lambda/(2*m) * sum(theta(2:end).^2); % don't regularize theta zero term
J = 1/(2*m) * sum((h - y).^2) + reg;

% Attention: must be careful of the matrix dimensions. 
% X is [12x2], first colum is 1, corresponds to theta0; X' is [2x12]
% X' * (h-y) is [2x1], corresponds to theta [2x1]
grad = 1/m * ( X' * (h-y) )+ lambda/m * theta;
grad(1) = 1/m * ( X' * (h-y) )(1);

% another method: 
%grad(1) = 1/m * sum((h-y).* X)'(1);
%grad(2:end) = 1/m * sum((h-y).* X)'(2:end) + lambda/m * theta(2:end);







% =========================================================================

grad = grad(:);

end

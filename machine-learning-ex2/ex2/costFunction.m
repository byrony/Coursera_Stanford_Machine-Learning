function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% sum(A) equals to A' * ones(size(A)(1), 1); which is vector of #rows x 1. 
% size(A) equals to size(A)(1)
% Attention: y is a column, log(sigmoid(X*theta) is also a column
J = (1/m) * ( (-y.*log(sigmoid(X*theta)) - (1-y).*log(1-sigmoid(X*theta)))' ...
* ones(size(y)(1),1));
%J = (1/m) * sum( (-y.*log(sigmoid(X*theta)) - (1-y).*log(1-sigmoid(X*theta))) );

grad = (1/m) * ((sigmoid(X*theta) - y)' * X);

% notes: cause in ex2: fprintf(' %f \n', grad); so no matter grad is a 1xN vector
% or Nx1 vector; it will print each value of vector line by line.  

% another code:
%h = sigmoid(X * theta);
%% h = [n x 1]
%
%costPos = -y' * log(h);
%costNeg = (1 - y') * log(1 - h);
%
%J = (1/m) * (costPos - costNeg);
%
%grad = (1/m) * (X' * (h - y));


% =============================================================

end

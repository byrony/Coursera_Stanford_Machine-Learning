function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% a trick works in Octave, for Matlab have to split it into two statements?
%y = [4 2 3 1]'               % just a vector for testing
%y_matrix = eye(4)(y,:)       % uses 'y' as an index to select a row of eye()

% Part 1
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
% y has size 5000 x 1
y_matrix = eye(num_labels)(y, :);  % y_matrix is [5000x10]
a1 = [ones(size(X ,1), 1) X];  %[5000 x 401]
a2 = sigmoid(a1 * Theta1'); %[5000 x 25]
h = sigmoid([ones(size(a2,1), 1) a2] * Theta2'); %[5000 x 26]x[26 x 10]=[5000x10]
%h = a3;
temp = y_matrix.* log(h) + (1-y_matrix).* log(1-h); % 
J = -1/m * sum(sum(temp)); % or sum(temp(:))

  % Part 1: regularized cost function.
Theta1_filter = Theta1(:, 2:end); % [25x400]
Theta2_filter = Theta2(:, 2:end); % [10x25]
reg = lambda/(2*m) * ( sum(sum(Theta1_filter.^2)) + sum(sum(Theta2_filter.^2)) );
J = J + reg;



% Part 2
D2 = 0;
D1 = 0;
for t = 1:m,
  a1 = [1 X(t, :)]'; % [401 x 1]
  z2 = Theta1*a1; % [25x401] * [401x1] = [25x1]
  a2 = [1 ; sigmoid(z2)]; %[26x1]
  z3 = Theta2*a2; % [10x26] * [26x1] = [10x1]
  a3 = sigmoid(z3); % [10x1], a3 is the output layer, do not need add 1.
  delta3 = a3-y_matrix(t, :)'; % [10x1]
  delta2 = Theta2_filter' * delta3 .* sigmoidGradient(z2); % [25x1] cause delta2
  % is a3-y3, which doesn't contain bias unit.
  
  D2 = D2 + delta3*a2'; % [10x26]
  D1 = D1 + delta2*a1'; % [25x401]
end
Theta2_grad = 1/m * D2;
Theta1_grad = 1/m * D1;

% Regularized. First add regularization to all units, then correct the 1st column
Theta2_grad = 1/m * D2 + lambda/m * Theta2; %[10x26]
Theta2_grad(:,1) = 1/m*D2(:,1);
Theta1_grad = 1/m * D1 + lambda/m * Theta1; %[25x401]
Theta1_grad(:,1) = 1/m*D1(:,1);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

Error = X * Theta' - Y;  # [nm x nu] = [nm x nf] * [nu x nf]'; Y is [nm x nu]
ErrorSq = Error.*Error;

# sum(sum(R.*M)) is the sum of all the elements of M for which the corresponding element in R equals 1
J = 1/2 * sum(sum(ErrorSq .* R)); 
J = J + lambda/2 * sum(sum(Theta.^2)) + lambda/2 *sum(sum(X.^2)); 

# when performing vectorization, check the Tips in ex8. It is clear.
# actually we didn't compute the derivative of each x and theta,
# but compute the first element of derivative vector for all the movies or users;
# then second, then third... 
X_grad = (Error.* R ) * Theta + lambda*X; # X_grad is [nm x nf] = [nm x nu] * [nu x nf]; X is [nm x nf]
Theta_grad = (Error .* R)' * X + lambda*Theta; # Theta_grad is [nu x nf] = [nm x nu]' * [nm x nf]; Theta is [nu x nf]











% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end

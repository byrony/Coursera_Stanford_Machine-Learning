function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

U_reduce = U(:, 1:K);
Z = U_reduce' * X'; 
# [K x n] * [n x 50] = [K x 50], 50 is # of sample data, # of rows of X
# each column is s new dimension of k
# use matrix vectorized implementation to avoid a loop of X
Z = Z'; #[50 x K], each column is a new dimension.

# notice is size of Z, should be the same as X, each column is a dimension.
% =============================================================

end

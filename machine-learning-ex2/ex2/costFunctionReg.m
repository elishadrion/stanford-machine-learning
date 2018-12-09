function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


% ====================== J calcuation ======================
for i = 1:rows(X)
  h0 = sigmoid(sum(theta.*(X(i, :)')));
  term1 = (-y(i))*log(h0);
  term2 = (1-y(i))*log(1-h0);
  J += (term1-term2);
endfor
J = J/m;

% Compute the regularization bit of the cost function
theta_sum = 0;
%We avoid theta 0
for i = 2:rows(theta)
  theta_sum += theta(i)^2;
endfor
J += theta_sum*(lambda/(2*m));

% ====================== Gradient calculation ======================

%compute theta 0 separately

total = 0;
for i = 1:rows(X)
  h0 = sigmoid(sum(theta.*(X(i, :)')));
  total += (h0-y(i))*X(i, 1);
endfor
grad(1) = total/m;

% compute the rest
for j = 2:rows(grad)
  total = 0;
  for i = 1:rows(X)
    h0 = sigmoid(sum(theta.*(X(i, :)')));
    total += (h0-y(i))*X(i, j);
  endfor
  total /= m;
  grad(j) = total + ((lambda/m)*theta(j));
endfor

% =============================================================

end

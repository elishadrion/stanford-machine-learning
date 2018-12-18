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


%X is the input layer, A1
%add A(1,0) the biais unit
X = [ones(m, 1) X];

%hidden layer
%All the z2 of our entire dataset, size of (25+1) x 5000
%The i-th column is the z2 for i-th example
all_z2 = Theta1*X';
all_A2 = sigmoid(all_z2);
%add A(2,0), the biais unit (as a row since our activation nodes are rows)
all_A2 = [ones(1, columns(all_A2)); all_A2];

%output layer
all_z3 = Theta2*all_A2;
all_A3 = sigmoid(all_z3);

% compute J
for i = 1:m
  %We "vectorized" the inner sum for tge K classes
  %h0(x_i) represents the output values as a matrix for the example i
  h0 = all_A3(:,i);
  yi = zeros(num_labels,1);
  yi(y(i)) = 1;
  part1 = (-yi).*log(h0);
  part2 = (1-yi).*log(1-h0);
  J += (sum(part1-part2));
  % delta for output layer
  delta3 = h0 - yi;  %D3 is the size of the ouput so here 10x1
  %remove biais of delta2 , size of 25x1
  delta2 = ((Theta2'*delta3)(2:end).*sigmoidGradient(all_z2(:, i)));
  %vectorized implementation of the gradient accumulation
  Theta2_grad += delta3*(all_A2(:, i)');
  Theta1_grad += delta2*(X(i, :));
endfor
J /= m;

% ====================== REGULARIZATION ======================
%biais is the first column, we exclude it
J += (sum(sum(Theta2(:, 2:end).^2))+sum(sum(Theta1(:, 2:end).^2)))*(lambda/(2*m));

%regularization of the gradient
Theta2_grad(:, 2:end) += lambda*Theta2(:, 2:end);
Theta1_grad(:, 2:end) += lambda*Theta1(:, 2:end);
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:)/m ; Theta2_grad(:)/m];


end

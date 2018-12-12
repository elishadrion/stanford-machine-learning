function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
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
[values p] = max(all_A3);

% =========================================================================


end

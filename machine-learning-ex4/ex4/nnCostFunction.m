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

% Theta1 has size 25 x 401, già calcolato
% Theta2 has size 10 x 26, già calcolato
X = [ones(m, 1) X]; % 5000x401
Z_2 = X * Theta1'; % 5000 x 25
A_2 = sigmoid(Z_2); % 5000 x 25
A_2 = [ones(m, 1) A_2]; % 5000 x 26
Z_3 = A_2 * Theta2'; % 5000 x 10
H = sigmoid(Z_3); % 5000 x 10
% [val, p] = max(H, [], 2); % 5000 x 1
delta_3 = zeros(m, num_labels); % 5000 x 10
delta_2 = zeros(m, hidden_layer_size); % 5000 x 25

for c = 1:num_labels
    y_nn = [1:num_labels] == y; % trasforma la label in vett di 0 e un 1 al posto giusto
    J = sum(sum( -y_nn .* log(H) - (1 - y_nn) .* log(1 - H) ) / m);
    delta_3 = H - y_nn;
end
disp('--- y_nn ---');
disp(size(y_nn));

d = (delta_3 * Theta2); % 5000 x 10 * 10 x 26
d = d(:, 2:end); % 5000 x 25
delta_2 = d .* sigmoidGradient(Z_2); % 5000 x 25

Theta1_grad = (delta_2' * X) ./ m; % 25 x 5000 * 5000 x 401
Theta2_grad = (delta_3' * A_2) ./ m; % 10 x 5000 * 5000 x 26

% ======== Regolarizzazione ========
ThetaReg1 = Theta1;
ThetaReg1(:, 1) = 0;
%disp('--- Theta1(1:5, 1) ---');
%disp(Theta1(1:5, 1));
%disp('--- ThetaReg1(1:5, 1) ---');
%disp(ThetaReg1(1:5, 1));
ThetaReg2 = Theta2;
ThetaReg2(:, 1) = 0;

disp('--- J non regolarizzato ---');
disp(J);

thetazza = (sum(sum(ThetaReg1.^2)) + sum(sum(ThetaReg2.^2)));

Theta1_grad = Theta1_grad + lambda / m .* ThetaReg1;
Theta2_grad = Theta2_grad + lambda / m .* ThetaReg2;

reg = lambda / (2*m) * thetazza;
disp('--- Size di thetazza ---');
disp(size(thetazza));
disp('--- thetazza ---');
disp(thetazza);
disp('--- lambda / (2*m) ---');
disp(lambda / (2*m));
disp('--- reg ---');
disp(reg);
J = J + reg;
disp('--- J regolarizzato ---');
disp(J);
% =========================================================================


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

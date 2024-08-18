function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
disp('Dimensione di m:')
disp(m)
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
disp('Size di theta:')
disp(size(theta))

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
thetaReg = theta;
thetaReg(1) = 0;

z = X * theta;
h = sigmoid(z);
disp('Size di X mappata:')
disp(size(X))
disp('Size di z:')
disp(size(z))
disp('Size di h:')
disp(size(h))

%J = sum( -y .* log(h) - (1 - y) .* log(1 - h) ) / m + lambda / (2*m) * (thetaReg' * thetaReg);
J = sum( -y .* log(h) - (1 - y) .* log(1 - h) ) / m + lambda / (2*m) .* sum(thetaReg.^2);
grad = (X' * (h - y)) / m + lambda / m * thetaReg;

% =============================================================




end

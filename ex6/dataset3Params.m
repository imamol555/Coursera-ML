function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
 
% create a matrix with each row = [c,sigma,validation_error]
% we wil try 8 values for each c and 8 values for sigma
result_mat = zeros(8*8,3);

current_row = 0;

for tryC = [0.01 0.03 0.1 0.3 1 3 10 30]
	for trySigma = [0.01 0.03 0.1 0.3 1 3 10 30]
		current_row = current_row + 1;
		model= svmTrain(X, y, tryC, @(x1, x2) gaussianKernel(x1, x2, trySigma));
		predictions = svmPredict(model,Xval);
		pred_error = mean(double(predictions ~= yval));
		
		result_mat(current_row,:) = [tryC, trySigma, pred_error];
	end
end

%sort the result_mat along error column in increasing order
sorted_result = sortrows(result_mat,3);

C = sorted_result(1,1);
sigma = sorted_result(1,2);







% =========================================================================

end

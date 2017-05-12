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

		
C_List=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_List = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
min_error = inf;
optimal_c = optimal_sigma =0;

for c=1:size(C_List')
	c_train = C_List(c);
	for s=1:size(sigma_List')
		s_train = sigma_List(s);
		%fprintf('SVM Train with C, sigma = %f, %f', c_train, s_train);
		model= svmTrain(X, y, c_train, @(x1, x2) gaussianKernel(x1, x2, s_train));
		predictions = svmPredict(model, Xval);
		error = mean(double(predictions ~= yval));
		
		if(error < min_error)
			min_error = error;
			optimal_c = c_train;
			optimal_sigma = s_train;
			%fprintf('\n new min found C, sigma = %f, %f with error = %f', optimal_c, optimal_sigma, min_error);
		end
	end
end

C = optimal_c;
sigma = optimal_sigma;

% =========================================================================

end

function [C, sigma] = findBestParams(X, y, Xval, yval)
%findBestParams returns the optimal C and sigma, using a gaussian kernel
C = 0;
sigma = 0;
error = inf;
Cvec = [0.01 0.03 0.1 0.3 1 3 10 30];

for i = 1:length(Cvec)
  for j = 1:length(Cvec)
    tempC = Cvec(i);
    tempSigma = Cvec(j);
    %train using the test set 
    model = svmTrain(X, y, tempC, @(x1, x2) gaussianKernel(x1, x2, tempSigma));
    %predict with the cv set
    predictions = svmPredict(model, Xval);
    tempError = mean(double(predictions ~= yval));
    if tempError < error;
      error = tempError;
      C = tempC;
      sigma = tempSigma;
    endif
  endfor
endfor
end
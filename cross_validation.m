
% This function is used to generate K-fold cross-validation partitions.
%  ----------- input ------------------------
% n: total number of observations being partitioned
% fold : number of folds
% -------------------------------------------
% ------------ output -----------------------
% tr_ind: (n by fold) indicators for constructing the training sample partitions
% test_ind : (n by fold) indicators for constructing the testing sample partitions
% -------------------------------------------

function [tr_ind, test_ind] = cross_validation(n,fold)

% k-fold cross validation

tr_ind=logical(zeros(n,fold));
test_ind=logical(zeros(n,fold));

cv=cvpartition(n,'Kfold',fold);
for i=1:cv.NumTestSets
tr_ind(:,i)=cv.training(i);
test_ind(:,i)=cv.test(i);
end
end

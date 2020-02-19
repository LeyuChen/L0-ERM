
% This function computes the optimal penalty tuning value for the L0-penalized maximum score estimation 
% among a range of tuning values specified by the vector lamda_range via the cross-validation procedure.
% The indices for the observations to be included in the training and testing 
% sample in the CV folds are specified by tr_ind and test_ind, respectively.

function [best_lamda, bhat,score,gap,rtime,ncount]=cv_penalized_max_score(tr_ind,test_ind,data,beta0,lamda_range,T,tol,bnd)

k=size(data,2)-1;
lamda_num = length(lamda_range); 
fold=size(tr_ind,2);
score=zeros(fold,lamda_num);
gap=zeros(fold,lamda_num);
rtime=zeros(fold,lamda_num);
ncount=zeros(fold,lamda_num);
bhat=zeros(k-1,fold,lamda_num);
val_score=zeros(lamda_num,1);

for q=1:lamda_num
 for i=1:fold
     disp(['(lamda,fold) : ' num2str(lamda_range(q)) ' ' num2str(i)]);
y=data(tr_ind(:,i),1);
datax=data(tr_ind(:,i),2:end);

[bhat(:,i,q),score(i,q),gap(i,q),rtime(i,q),ncount(i,q)]  = penalized_max_score_fn(y,datax(:,1),datax(:,2:end),beta0,lamda_range(q),T,tol,bnd);  

y_v=data(test_ind(:,i),1);
datax_v=data(test_ind(:,i),2:end);
val_score(q) = val_score(q)+ mean(y_v == ((datax_v*[beta0;bhat(:,i,q)])>=0)); 
 end
 val_score(q)=val_score(q)/fold;
end
[~,best_lamda] = max(val_score);
end
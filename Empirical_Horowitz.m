
% These codes can be used to reproduce the empirical results reported in
% Table 3 of the paper. The results are concerned with the prediction of
% transportation mode choice based on the dataset of Horowitz (1993).


clear;

% set method = 1 to reproduce the results using logit_lasso
% set method = 2 to reproduce the results using subset selection
% set method = 3 to reproduce the results using L0-ERM

method = 1; % 1 for logit_lasso
            % 2 for subset selection
            % 3 for L0-ERM

folder = pwd;
folder1 = [folder '\glmnet_matlab']; addpath(folder1);
load('horowitz_data.mat'); % load the work-trip mode choice dataset
% data columins : [Y DCOST CARS DOVTT DIVTT]

rng(100,'twister');   % fix the random seed 

[n,k]=size(data);
z2 = data(:,3); z3 = data(:,4); z4 = data(:,5);
y=data(:,1); focus_x=data(:,2);
x_aux1 = data(:,3:end); % linear terms
x_aux2 = [z2.*z3 z3.*z4 z2.*z4]; % cross-product terms
x_aux3 = [z2.*z2 z3.*z3 z4.*z4]; % quadratic terms

x_aux4 = [z2.*z2.*z2 z3.*z3.*z3 z4.*z4.*z4]; % cubic terms
x_aux5 = [z2.*z2.*z3 z2.*z2.*z4 z3.*z3.*z2 z3.*z3.*z4 z4.*z4.*z2 z4.*z4.*z3];
x_aux6 = z2.*z3.*z4;
aux_x = [x_aux1 x_aux2 x_aux3 x_aux4 x_aux5 x_aux6];
clear x_aux1 x_aux2 x_aux3 x_aux4 x_aux5 x_aux6 z2 z3 z4;  

std_aux_x=zscore(aux_x,1); 
datax=[zscore(focus_x,1) ones(n,1) std_aux_x];
clear focus_x aux_x std_aux_x;

idx=randperm(n); % shuffle the observations
n_tr = round(n/3); % training sample size is about one third of the entire data observations   

y_tr = y(idx(1:n_tr)); % extract the training sample 
datax_tr = datax(idx(1:n_tr),:);

y_val = y(idx(n_tr+1:end)); % extract the validation sample
datax_val = datax(idx(n_tr+1:end),:);

p=size(datax,2)-1; beta0=1;

varname1 = ["DCOST" "Intercept" "CARS" "DOVTT" "DIVTT"]; 
varname2 = ["CARS*DOVTT" "DOVTT*DIVTT" "CARS*DIVTT"];
varname3 = ["CARS*CARS" "DOVTT*DOVTT" "DIVTT*DIVTT"];
varname4 = ["CARS*CARS*CARS" "DOVTT*DOVTT*DOVTT" "DIVTT*DIVTT*DIVTT"];
varname5 = ["CARS*CARS*DOVTT" "CARS*CARS*DIVTT" "DOVTT*DOVTT*CARS" "DOVTT*DOVTT*DIVTT"];
varname6 = ["DIVTT*DIVTT*CARS" "DIVTT*DIVTT*DOVTT" "CARS*DOVTT*DIVTT"];
varname = [varname1 varname2 varname3 varname4 varname5 varname6];


if method==1 % logit_lasso based prediction via glmnet

options = glmnetSet; % Extract default glmnet options
options.standardize=false;
options.penalty_factor=[0 ones(1,p)]; % Do not penalize the focused covariate
cvfit = cvglmnet(datax_tr(:,[1 3:end]), y_tr, 'binomial', options, 'class', 10);
bhat_glm=cvglmnetCoef(cvfit,'lambda_min');
bhatopt = [bhat_glm(2);bhat_glm(1);bhat_glm(3:end)];
bhat_glm=cvglmnetCoef(cvfit,'lambda_1se');
bhat1se = [bhat_glm(2);bhat_glm(1);bhat_glm(3:end)];

selopt=sum((abs(bhatopt(2:end))>1e-6)); 
sel1se=sum((abs(bhat1se(2:end))>1e-6));

disp(['estimated sparsity using optimal penalty tuning value (lamda_opt): ' num2str(selopt)]);
disp(['estimated sparsity using penalty tuning value based on one-standard-error rule (lamda_1se): ' num2str(sel1se)]);

in_riskopt=1-mean(y_tr == ((datax_tr*bhatopt)>=0)); 
in_risk1se=1-mean(y_tr == ((datax_tr*bhat1se)>=0)); 

disp(['in-sample risk using optimal penalty tuning value (lamda_opt): ' num2str(in_riskopt)]);
disp(['in-sample risk using penalty tuning value based on one-standard-error rule (lamda_1se): ' num2str(in_risk1se)]);

out_riskopt=1-mean(y_val == ((datax_val*bhatopt)>=0)); 
out_risk1se=1-mean(y_val == ((datax_val*bhat1se)>=0)); 

disp(['out-of-sample risk using optimal penalty tuning value (lamda_opt): ' num2str(out_riskopt)]);
disp(['out-of-sample risk using penalty tuning value based on one-standard-error rule (lamda_1se): ' num2str(out_risk1se)]);
disp('Scale normalized estimated coefficients:');
disp([varname' num2str([bhatopt/bhatopt(1) bhat1se/bhat1se(1)])]); % display scale normalized estimated coefficients

else % the following setups are for MIO based methods (subset selection or L0-ERM)

maxT=0; tol=0; 
bnd=[-10*ones(p,1) 10*ones(p,1)]; % parameter bounds

if method==2  % best subset selection based prediction via the MIO
cv_tuning=1:p; % subset sizes to be calibrated   

else % L0-ERM based prediction via the MIO
cv_tuning=[1/32;1/16;1/8;1/4;1/2;1;2]; % tuning multipliers to be calibrated
c_bar=log(log(max([n_tr;p])));
lamda_rate = sqrt(log(max([n_tr;p]))/n_tr);
end

lamda_num=length(cv_tuning);

fold=5; % tuning parameters are calibrated using 5-fold cross validation
[tr_ind, test_ind] = cross_validation(n_tr,fold);

bhat=zeros(p,1);
bhat_cv=zeros(p,fold,lamda_num);
gap_cv=zeros(fold,lamda_num);
rtime_cv=zeros(fold,lamda_num);
ncount_cv=zeros(fold,lamda_num);

try
if method==2 % subset selection method
[ind_lamda, bhat_cv,~,gap_cv,rtime_cv,ncount_cv]=cv_best_subset_maximum_score(tr_ind,test_ind,[y_tr datax_tr],1,2:size(datax,2),beta0,cv_tuning,maxT,tol,bnd);
best_q = cv_tuning(ind_lamda); % the calibrated optimal subset size
[bhat,score,gap,rtime,ncount]  = max_score_constr_fn(y_tr,datax_tr(:,1),datax_tr(:,2:end),beta0,best_q,maxT,tol,bnd);

else % L0-ERM method

% ---  These codes compute the lamda values being calibrated.  
[bhat,score,gap,rtime,ncount]  = max_score_constr_fn(y_tr,datax_tr(:,1:2),datax_tr(:,3:end),beta0,0,maxT,tol,bnd); 
v = mean(y_tr == ((datax_tr*[beta0;bhat])>=0)); 
c=c_bar*v*(1-v);
lamda=c*lamda_rate*cv_tuning;
% -----------------------------------------------------------

[ind_lamda, bhat_cv,~,gap_cv,rtime_cv,ncount_cv]=cv_penalized_max_score(tr_ind,test_ind,[y_tr datax_tr],beta0,lamda,maxT,tol,bnd);
best_lamda=lamda(ind_lamda); % the calibrated optimal penalty tuning value
[bhat,score,gap,rtime,ncount]  = penalized_max_score_fn(y_tr,datax_tr(:,1),datax_tr(:,2:end),beta0,best_lamda,maxT,tol,bnd); 

end
catch gurobiError
    fprintf('Error reported\n');
end

sel=sum((abs(bhat)>1e-6));
in_risk=1-mean(y_tr == ((datax_tr*[beta0;bhat])>=0)); 
out_risk=1-mean(y_val == ((datax_val*[beta0;bhat])>=0)); 

if method==2
disp("Estimation results using subset selection");
else
disp("Estimation results using L0-ERM");
end

disp(['estimated sparsity: ' num2str(sel)]);
disp(['in-sample risk: ' num2str(in_risk)]);
disp(['out-of-sample risk: ' num2str(out_risk)]);
disp('Estimated coefficients:');
disp([varname' num2str([beta0;bhat])]);

end

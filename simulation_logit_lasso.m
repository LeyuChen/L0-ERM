
function simulation_logit_lasso(N,R,beta,DGP,data_folder)

K=length(beta);
bhat=zeros(K,R);
rtime=zeros(R,1);
sel=zeros(R,1);
sel_all = zeros(R,1);
num_irrel=zeros(R,1);
num_sel=zeros(R,1);
DGP_score=zeros(R,1); % in-sample score at the DGP parameter vector
val_score=zeros(R,1); % in-sample score at the estimated parameter vector
DGP_score_test=zeros(R,1); % out-of-sample score at the DGP parameter vector
val_score_test=zeros(R,1); % out-of-sample score at the estimated parameter vector

bhat1se=zeros(K,R);
sel1se=zeros(R,1);
sel_all1se = zeros(R,1);
num_irrel1se=zeros(R,1);
num_sel1se=zeros(R,1);
val_score1se=zeros(R,1); % in-sample score at the estimated parameter vector
val_score_test1se=zeros(R,1); % out-of-sample score at the estimated parameter vector

aux = (beta(2:end) ~=0);
tol=1e-6;

options = glmnetSet; % Extract default glmnet options
options.standardize=false;
options.penalty_factor=[0 ones(1,K-1)]; % Do not penalize the focused covariate

for i=1:R
disp(['Simulation repetition ' num2str(i)]);        
data = load([pwd '\' data_folder '\data' num2str(i)],'data');
y=data.data(1:N,1);
datax=data.data(1:N,2:end);

timeval=tic;
cvfit = cvglmnet(datax(:,[1 3:end]), y, 'binomial', options, 'class', 10);
rtime(i)=toc(timeval);

bhat_glm=cvglmnetCoef(cvfit,'lambda_min');
bhat(:,i) = [bhat_glm(2);bhat_glm(1);bhat_glm(3:end)];
bhat_glm=cvglmnetCoef(cvfit,'lambda_1se');
bhat1se(:,i) = [bhat_glm(2);bhat_glm(1);bhat_glm(3:end)];
    
 if abs(bhat(3,i))>tol
 sel(i) = 1;
 end
 
 if abs(bhat1se(3,i))>tol
 sel1se(i) = 1;
 end

 if aux == (abs(bhat(2:end,i))>tol)
sel_all(i) = 1;    
 end
 
 if aux == (abs(bhat1se(2:end,i))>tol)
sel_all1se(i) = 1;    
 end
 
  num_irrel(i) = sum([(abs(bhat(2,i))>tol);abs(bhat(4:end,i))>tol]);
  num_irrel1se(i) = sum([(abs(bhat1se(2,i))>tol);abs(bhat1se(4:end,i))>tol]);
  
  num_sel(i)=sum(abs(bhat(2:end,i))>tol);
  num_sel1se(i)=sum(abs(bhat1se(2:end,i))>tol);
  
DGP_score(i) = mean(y == ((datax*beta)>=0)); 
val_score(i) = mean(y == ((datax*bhat(:,i))>=0)); 
val_score1se(i) = mean(y == ((datax*bhat1se(:,i))>=0)); 

  y_val=data.data(N+1:end,1);
  datax_val=data.data(N+1:end,2:end);
  DGP_score_test(i) = mean(y_val == ((datax_val*beta)>=0)); 
  val_score_test(i) = mean(y_val == ((datax_val*bhat(:,i))>=0)); 
  val_score_test1se(i) = mean(y_val == ((datax_val*bhat1se(:,i))>=0)); 
  %}  
end

%rmdir(data_folder,'s'); % activate this to delete the saved simulation data

output_str = ["Corr_sel";"Orac_sel";"Num_irrel";"in_RR";"out_RR"];

disp(['Simulation results for logit_lasso under DGP ' DGP ' with p = ' num2str(K-1)]);
disp(['Results using optimal penalty tuning value (lamda_opt):']);
disp([output_str num2str((mean([sel sel_all num_irrel (1-val_score)./(1-DGP_score) (1-val_score_test)./(1-DGP_score_test)]))')]);
disp(['Results using penalty tuning value based on one-standard-error rule (lamda_1se):']);
disp([output_str num2str((mean([sel1se sel_all1se num_irrel1se (1-val_score1se)./(1-DGP_score) (1-val_score_test1se)./(1-DGP_score_test)]))')]);
end



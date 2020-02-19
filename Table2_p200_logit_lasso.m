
% These codes can be used to replicate the logit_lasso simulation results
% reported in Table 2 for the case with p=200.

clear;

folder = pwd;
folder1 = [folder '\glmnet_matlab']; addpath(folder1);

rng(1,'twister'); rng_state = rng;

p=200; type=1; 
data_folder = ['Table' num2str(type+1) '_p' num2str(p)];

R = 100; % simulation repetitions
N = 100; % size of the training sample
N_val = 5000; % size of the validation sample

if type==1
beta_s = -1.85; DGP = '(ii)'; % heteroskedastic error design
else
beta_s = -0.55; DGP = '(i)'; % homoskedastic error design
end

beta0=1;
beta = [beta0; 0; beta_s; zeros(p-2,1)];
K=length(beta);

rho=0.25;
sigma=ones(K-1,1);
for i=1:K-2
sigma(i+1)=rho^i;
end
sigma=toeplitz(sigma);

save_simulation_data(N,N_val,R,beta,sigma,type,data_folder); % generate and save the simulation data 

rng(rng_state); 

simulation_logit_lasso(N,R,beta,DGP,data_folder); 


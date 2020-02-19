
function simulation_ERM(p,type)

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
aux = (beta(2:end) ~=0); tol=1e-6;

rho=0.25;
sigma=ones(K-1,1);
for i=1:K-2
sigma(i+1)=rho^i;
end
sigma=toeplitz(sigma);

bhat=zeros(K-1,R);
gap=zeros(R,1); % MIO gap
rtime=zeros(R,1); % MIO running time
ncount=zeros(R,1); % MIO node count
score=zeros(R,1); % MIO score

sel = zeros(R,1);
sel_all = zeros(R,1);
num_irrel=zeros(R,1);
num_sel=zeros(R,1);

DGP_score=zeros(R,1); % in-sample score at the DGP parameter vector
val_score=zeros(R,1); % in-sample score at the estimated parameter vector
DGP_score_test=zeros(R,1); % out-of-sample score at the DGP parameter vector
val_score_test=zeros(R,1); % out-of-sample score at the estimated parameter vector

bnd=[-10*ones(size(bhat,1),1) 10*ones(size(bhat,1),1)];

maxT=3600; abgap=0;

c_bar=log(log(max([N;p])));
lamda_rate = sqrt(log(max([N;p]))/N);

for i=1:R
disp(['Simulation repetition ' num2str(i)]);        
[y,datax] = simulation_data(N,beta,sigma,type);

try

[bhat(:,i),score(i),gap(i),rtime(i),ncount(i)]  = max_score_constr_fn(y,datax(:,1:2),datax(:,3:end),beta0,0,maxT,abgap,bnd);
v = mean(y == ((datax*[beta0;bhat(:,i)])>=0));    
c=c_bar*v*(1-v);
lamda=c*lamda_rate;

[bhat(:,i),score(i),gap(i),rtime(i),ncount(i)]  = penalized_max_score_fn(y,datax(:,1),datax(:,2:end),beta0,lamda,maxT,abgap,bnd);  

catch gurobiError
    fprintf('Error reported\n');
end

if abs(bhat(2,i))>tol
sel(i) = 1;
end

if aux == (abs(bhat(:,i))>tol)
sel_all(i) = 1;    
end
 
num_irrel(i) = sum([(abs(bhat(1,i))>tol);(abs(bhat(3:end,i))>tol)]);
num_sel(i)=sum(abs(bhat(:,i))>tol);

DGP_score(i) = mean(y == ((datax*beta)>=0)); 
val_score(i) = mean(y == ((datax*[beta0;bhat(:,i)])>=0)); 

if N_val>0
  [y_val,datax_val] = simulation_data(N_val,beta,sigma,type);
  DGP_score_test(i) = mean(y_val == ((datax_val*beta)>=0)); 
  val_score_test(i) = mean(y_val == ((datax_val*[beta0;bhat(:,i)])>=0)); 
 end

end

output_str = ["Corr_sel";"Orac_sel";"Num_irrel";"in_RR";"out_RR"];
disp(['Simulation results for L0-ERM under DGP ' DGP ' with p = ' num2str(p)]);
disp([output_str num2str((mean([sel sel_all num_irrel (1-val_score)./(1-DGP_score) (1-val_score_test)./(1-DGP_score_test)]))')]);

end


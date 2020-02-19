
% function input :
% y : vector of binary outcomes
% x_foc : (n by k) matrix of data for focused covariates, 
%         of which the first column should contain data of 
%         the contiuous regressor with respect to which 
%         scale normalization is imposed
% x_aux : (n by d) matrix of data for auxiliary covariates which will be
%         selected based on the penalized estimation procedure
% beta0 : the coefficient taking value either 1 or -1 to normalize the 
%         scale for the first covariate in x_foc       
% lamda : the tuning parameter for the model penalty
% T     : the time limit specified for early termination of the MIO solver
% bnd   : (((k-1)+d) by 2) matrix where the first and second columns  
%         respectively store the lower and upper bounds 
%         of the unknown coefficients
%         the first (k-1) rows correspond to the bounds of the focused 
%         covariates excluding the first one. 
%         the remaining d rows correspond to the bounds of the auxiliary
%         covariates

% function output :
% bhat  : the maximum score estimates for the unknown coefficients
% score : the value of maximum score objective function
% gap   : the MIO optimization gap value in case of early termination
%         gap = 0 ==> optimal solution is found within the time limit
% rtime : the time used by the MIO solver in the estimation procedure

function [bhat,score,gap,rtime,ncount] = penalized_max_score_fn(y,x_foc,x_aux,beta0,lamda,T,abgap,bnd)


N=length(y);
k=size(x_foc,2)-1;
d=size(x_aux,2);

bhat=zeros(k+d,1);
gap=0;
rtime=0;

miobnd=miobnd_fn([x_foc x_aux],beta0,bnd);

model.sense = '<';
model.modelsense = 'max';

model.lb = [zeros(N,1); bnd(:,1); zeros(d,1)];
model.ub = [ones(N,1); bnd(:,2); ones(d,1)];

% 'B' : int code 66
% 'C' : int code 67
model.vtype = char([66*ones(1,N) 67*ones(1,k+d) 66*ones(1,d)]); 

tol=1e-6;

params.outputflag = 0; 
params.OptimalityTol=tol;
params.FeasibilityTol=tol;
params.IntFeasTol=tol;

if T > 0
params.TimeLimit =T;
end

if abgap > 0
params.MIPGapAbs=abgap;
end

ztemp1=zeros(N,d);
htemp=[eye(d);-eye(d)];
etemp=[-diag(bnd(k+1:k+d,2));diag(bnd(k+1:k+d,1))];
mtemp1=[zeros(2*d,N+k) htemp etemp];

model.obj = [(2*y-1)/N;zeros(k+d,1);-lamda*ones(d,1)]; 
model.objcon = mean(1-y);
miobnd_bar = miobnd+tol;
mtemp2=[diag(-miobnd_bar) x_foc(:,2:k+1) x_aux ztemp1];
model.A = sparse([diag(miobnd) -x_foc(:,2:k+1) -x_aux ztemp1;mtemp2;mtemp1]);
model.rhs = [miobnd*(1-tol)+beta0*x_foc(:,1);-tol*miobnd_bar-beta0*x_foc(:,1);zeros(2*d,1)];

try
    result = gurobi(model, params);
    bhat=result.x(N+1:N+k+d);
    score=result.objval;
    gap=(result.objbound-result.objval);
    rtime=result.runtime;
    ncount=result.nodecount;
    
    fprintf('Optimization returned status: %s\n', result.status);
  
catch gurobiError
    fprintf('Error reported\n');
end

end


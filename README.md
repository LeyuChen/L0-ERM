Matlab codes for replicating both the simulation and empirical results of Chen and Lee (2020) on binary classification through the L0-penalized empirical risk minimization (ERM) approach. Description of the L0-penalized ERM and details of its numerical studies can be found in the paper:

Chen, Le-Yu and Lee, Sokbae (2020), "Binary Classification with Covariate Selection through L0-Penalized Empirical Risk Minimization".

The latest working paper version of this work can be found in this repository.

The empirical study is based on the transportation mode choice dataset of Horowitz (1993). This dataset is stored as a Matlab data matrix in the file "horowitz_data.mat". A plain ASCII version for this dataset is also provided in the file "worktrip".

Here is a description of the variables in this dataset.

Y: an indicator of whether automobile is the chosen mode of transportation
DCOST : the transit fare minus automobile travel cost in dollars
CARS  : the number of cars owned by the traveler's household
DOVTT : the transit out-of-vehicle travel time minus automobile out-of-vehicle travel time in minutes
DIVTT : the transit in-vehicle travel time minus automobile in-vehicle travel time in minutes
 
To replicate the simulation and empirical results of the paper, put all program and data files in the same work directory. 

The Matlab version of the Gurobi solver has to be installed for running the codes for solving the MIO based ERM problems. The Gurobi solver is freely available for academic purposes. You can download it via:

https://www.gurobi.com/

Since the ERM problem admits multiple global minimizers, use of different Gurobi versions may yield different classifier coefficient estimates. But the minimized objective values (in-sample risk) would be essentially identical. 

Note that the problem of minimizing empirical misclassification risk is equivalent to that of the maximum score estimation. Therefore, the ERM implementation here is based on the MIO formulation of maximum score estimation problem. 

Replication of the logit_lasso estimation results requires the Matlab implementation of the glmnet algorithms. The glmnet codes, which were written by Qian, Hastie, Friedman, Tibshirani, and Simon (2013), are collected in the folder "glmnet_matlab". 

To replicate Tables 1 and 2 of the paper, simply run the following Matlab programs:

Table1_p10_ERM.m
Table1_p10_logit_lasso.m
Table1_p200_ERM.m
Table1_p200_logit_lasso.m

Table2_p10_ERM.m
Table2_p10_logit_lasso.m
Table2_p200_ERM.m
Table2_p200_logit_lasso.m


Note that for the simulations with p = 200, we set a time limit of one hour for each MIO computation. Simulations for this case could take a long time as there are 100 repetitions, each of which could take up to one hour.


To replicate Table 3 of the paper, simply run the program "Empirical_Horowitz.m" with the setting:

% set method = 1 to reproduce the results using logit_lasso
% set method = 2 to reproduce the results using subset selection
% set method = 3 to reproduce the results using L0-ERM





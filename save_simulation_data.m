
% This function is used to generate simulation data which will be saved
% in the specified folder.

function save_simulation_data(N,N_val,R,beta,sigma,type,folder)

mkdir(folder); K=length(beta);

data = zeros(N+N_val,K+1);

for i=1:R
  [y,datax] = simulation_data(N,beta,sigma,type);
  data(1:N,:)=[y datax];
  [y_val,datax_val] = simulation_data(N_val,beta,sigma,type);
  data(N+1:end,:)=[y_val datax_val];
  save([pwd '\' folder '\data' num2str(i)],'data'); % save simulation data in the specified folder.
end
end
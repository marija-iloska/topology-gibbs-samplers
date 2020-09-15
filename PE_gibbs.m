function [fscore, MSE] = PE_gibbs(true_vals, init_vals, params, settings)

% Assign parameters
[term1, logpidet, log_rho1, log_rho0, I, I0, K, T, dim_y, dim_x] = settings{:};
[y, C, A] = true_vals{:};
[~, A_samples] = init_vals{:};
[var_u, ~, sig_0, mu_0] = params{:};


% Get MAP 
% Obtain the posterior of the vectorized matrix 
[mu_c, ~, ~, ~] = mn_conjugate_var(y, var_u, mu_0, sig_0);

% Convert to an estimate of the coefficient matrix
C_est = reshape(mu_c, dim_y, dim_y);


% GIBBS SAMPLER 
for i=1:I
    
    % Sampling Ajk_________________________________________________
    for j=1:dim_y
        for k = 1:dim_y
            % Case Ajk = 1
            A_samples(j,k) = 1;
            C_temp = C_est.*A_samples;
            
            % Compute terms in exponent
            term2 = 2*sum(sum(y(:,2:T).*(C_temp*y(:,1:T-1))));
            term3 = sum(sum((C_temp*y(:,1:T-1)).^2));
            
            % Compute log likelihood
            log_pa1 = log_rho1 + logpidet - 0.5*( term1 - term2 + term3 )/var_u;
            
            
            % Case Ajk = 0
            A_samples(j,k) = 0;
            C_temp = C_est.*A_samples;
            
            % Compute terms in exponent
            term2 = 2*sum(sum(y(:,2:T).*(C_temp*y(:,1:T-1))));
            term3 = sum(sum((C_temp*y(:,1:T-1)).^2));
            
            % Compute log likelihood
            log_pa0 = log_rho0 + logpidet - 0.5*( term1 - term2 + term3 )/var_u;
            
            
            
            % Scale them
            pa0 = exp(log_pa0 - max([log_pa0, log_pa1]));
            pa1 = exp(log_pa1 - max([log_pa0, log_pa1]));
            
            % Normalize
            prob1 = pa1/(pa1+pa0);
            
            % Sample Ajk
            A_samples(j,k) = rand < prob1;
        end
    end
    
    A_store(:,:,i) = A_samples;
    
end


% Apply burn-in
A_store = A_store(:,:, I0+1:K:I);

% Get estimate
A_hat = mode(A_store,3);

% Compute fscore
[~, ~, fscore] = adj_eval(A, A_hat);

% Compute MSE in the coefficients
MSE = sum(sum((C-C_est).^2))/dim_x;

end

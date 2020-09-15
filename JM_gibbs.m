function [fscore, MSE] = JM_gibbs(true_vals, init_vals, params, settings)

% Assign parameters
[term1, logpidet, log_rho1, log_rho0, I, I0, K, T, dim_y, dim_x] = settings{:};
[y, C, A] = true_vals{:};
[~, A_samples] = init_vals{:};
[var_u, ~, sig_0, mu_0] = params{:};


% Obtain the posterior of the vectorized matrix
[mu_c, sig_c, ~, ~] = mn_conjugate_var(y, var_u, mu_0, sig_0);

for i=1:I
    

    % Sampling Cs_________________________________________________
    C_samples = mvnrnd(mu_c, sig_c);
    % Convert to an estimate of the coefficient matrix
    C_samples = reshape(C_samples, dim_y, dim_y);
    
    
    % Sampling Ajk_________________________________________________
    for j=1:dim_y
        for k = 1:dim_y
            % Case Ajk = 1
            A_samples(j,k) = 1;
            C_temp = C_samples.*A_samples;
            
            % Compute terms in exponent
            term2 = 2*sum(sum(y(:,2:T).*(C_temp*y(:,1:T-1))));
            term3 = sum(sum((C_temp*y(:,1:T-1)).^2));
            
            % Compute log likelihood
            log_pa1 = log_rho1 + logpidet - 0.5*( term1 - term2 + term3 )/var_u;           
            
            
            
            % Case Ajk = 0
            A_samples(j,k) = 0;
            C_temp = C_samples.*A_samples;
            
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
    C_store(:,:,i) = C_samples;
    
end


% Apply burn-in
A_store = A_store(:,:, I0+1:K:I);
C_store = C_store(:,:, I0+1:K:I);

A_hat = mode(A_store,3);
C_hat = mean(C_store,3);

% Hadammard product
C_hat = A_hat.*C_hat;

[~, ~, fscore] = adj_eval(A, A_hat);
MSE = sum(sum((C-C_hat).^2))/dim_x;


end
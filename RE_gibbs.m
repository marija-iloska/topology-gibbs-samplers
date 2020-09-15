function [fscore, MSE] = RE_gibbs(true_vals, init_vals, params, settings)

% Assign parameters
[term1, logpidet, log_rho1, log_rho0, I, I0, K, T, dim_y, dim_x] = settings{:};
[y, C, A] = true_vals{:};
[C_samples, A_samples] = init_vals{:};
[var_u, var_0, ~, ~] = params{:};


for i=1:I
    % Initialize
    
    % Sampling a_jk, and c_jk
    for j=1:dim_y
        for k=1:dim_y

            % Sampling Ajk_________________________________________________
            
            
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
            
            % Sampling Cjk_________________________________________________
            
            
            % Product of likelihoods           
            var_jk = var_u/sum(y(k,1:T-1).^2);  
            mu_jk = var_jk/var_u;
            
            t1 = sum(y(j,2:T).*y(k,1:T-1));
            c1 = sum(C_samples(j,:)'.*y(:,1:T-1),1);        
            t2 = sum(  y(k,1:T-1).*( c1 - C_samples(j,k)*y(k,1:T-1) )  );
            
            mu_jk = mu_jk* (t1 - t2);
            
                     
            if (A_samples(j,k) ==1)
                % Posterior of Cjk
                var_post = var_jk*var_0/(var_jk + var_0);
                mu_post = var_post*mu_jk/var_jk;
                
                % Sample Cjk from posterior
                C_samples(j,k) = normrnd(mu_post, var_post);
            else
                % Sample Cjk from prior
                C_samples(j,k) = normrnd(0, var_0);
            end
            
        end
    end
    
    % Store sample from iteration i
    A_store(:,:,i) = A_samples;
    C_store(:,:,i) = C_samples;
end

% Apply burn-in
A_store = A_store(:,:, I0+1:K:I);
C_store = C_store(:,:, I0+1:K:I);

% Find mode and mean
A_hat = mode(A_store,3);
C_hat = mean(C_store,3);

% Hadammard product
C_hat = A_hat.*C_hat;

[~, ~, fscore] = adj_eval(A, A_hat);
MSE = sum(sum((C-C_hat).^2))/dim_x;


end


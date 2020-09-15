function [fscore, MSE] = RM_gibbs(true_vals, init_vals, params, settings)

% Assign parameters
[~, ~, log_rho1, log_rho0, I, I0, K, T, dim_y, ~] = settings{:};
[y, C, A] = true_vals{:};
[C_old, A_old] = init_vals{:};
[var_q, var_c,~, ~] = params{:};


%% Particle GIBBS Loop
for i=1:I
      
    % SAMPLING C~__________________________________________________________
    for j = 1:dim_y
        [mu_c, sig_c] = compute_c(y, A_old, var_c, var_q,j, T, dim_y);
        C_old(j,:) = mvnrnd(mu_c,sig_c);
    end
    
    
    % SAMPLING A___________________________________________________________
    [A_old] = compute_a(y, A_old, C_old, log_rho0, log_rho1, var_q, T, dim_y);
    
    % SAMPLING C___________________________________________________________
    C_eff = A_old.*C_old;
      
    
    % STORING______________________________________________________________
    %C_store(:,:,i) = C_old;
    Adj(:,:,i) = A_old;
    Ceff(:,:,i) = C_eff;
    
end

% Apply the burn in
A_samples = Adj(:,:,I0+1:K:I);
C_samples = Ceff(:,:,I0+1:K:I);

% Find mode and mean
A_hat = mode(A_samples,3);
C_hat = mean(C_samples,3);

% Hadammard product
C_hat0 = A_hat.*C_hat;

[~, ~, fscore] = adj_eval(A, A_hat);
MSE = sum(sum((C-C_hat0).^2))/(dim_y^2);


end
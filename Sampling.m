clear all
clc

%parpool(34)

tic
R = 1;
for run = 1:R
    tic
    %% Generating data
    dim_y = 5; var_u =1;
    p_s = 0.7; p_ns = 0.3;
    T = 1000;
    
    % Generating data and matrices
    [A, C, y, dim_x] = generate_mat(T, dim_y, p_s, p_ns, var_u);
    
    
    %Consider a Gaussian prior
    mu_0 = zeros(dim_x, 1);
    var_0 = 1; sig_0 = var_0*eye(dim_x);
    
    
    %% Gibbs Settings
    
    % Settings for Gibbs Bernoulli
    I = 5000;                       % Gibbs iterations
    I0 = 2500;                       % Gibbs burn-in
    K = 2;                          % Thinning parameter
    
    
    % Initial adjacency and coefficient matrix
    A_samples = ones(dim_y, dim_y);
    C_samples = reshape(mvnrnd(mu_0, sig_0), dim_y, dim_y);
    
    % Assume Bernoulli prior for a_jk for now with p(a_jk=1) = 0.4
    rho = 0.4;
    
    % Common terms used in all samplers; Avoiding redundant computation
    log_rho1 = log(rho);
    log_rho0 = log(1-rho);
    logpidet = -0.5*dim_y*log(2*pi*det(var_u*eye(dim_y)));
    term1 = sum(sum(y(:,2:T).^2,1));
    
    
    % Store parameters in a cell
    settings = {term1, logpidet, log_rho1, log_rho0, I, I0, K, T, dim_y, dim_x};
    true_vals = {y, C, A};
    init_vals = {C_samples, A_samples};
    params = {var_u, var_0, sig_0, mu_0};
    
    
    
    %% SECTION 1 Samplers
    
    % Regular Sampling by Element
    [fscore0, MSE0] = RE_gibbs(true_vals, init_vals, params, settings);
    f_re(run) = fscore0;
    mse0(run) = MSE0;
    
    % Regular Sampling by Matrix
    [fscore1,  MSE1] = RM_gibbs(true_vals, init_vals, params, settings);
    f_rm(run) = fscore1;
    mse1(run) = MSE1;
    
    
    % _______________________________________________________________________
    % Joint Sampling by Element
    [fscore2, MSE2] = JE_gibbs(true_vals, init_vals, params, settings);
    f_je(run) = fscore2;
    mse2(run) = MSE2;
    
    % Joint Sampling by Matrix
    [fscore3, MSE3] = JM_gibbs(true_vals, init_vals, params, settings);
    f_jm(run) = fscore3;
    mse3(run) = MSE3;
    
    
    % _______________________________________________________________________
    % Point estimate
    [fscore4, MSE4] = PE_gibbs(true_vals, init_vals, params, settings);
    f_pe(run) = fscore4;
    mse4(run) = MSE4;
    
    
    %% SECTION 2 Samplers
    
    % Regular Sampling by Element reversed
    [fscore5, MSE5] = REr_gibbs(true_vals, init_vals, params, settings);
    f_rer(run) = fscore5;
    mse5(run) = MSE5;
    
    % Regular Sampling by Matrix reversed
    [fscore6, MSE6] = RMr_gibbs(true_vals, init_vals, params, settings);
    f_rmr(run) = fscore6;
    mse6(run) = MSE6;
    
    
    toc
end
toc

fs_RE = mean(f_re);
mse_RE = mean(mse0);

fs_RM = mean(f_rm);
mse_RM =mean(mse1);

fs_JE = mean(f_jm);
mse_JE = mean(mse2);

fs_JM = mean(f_je);
mse_JM = mean(mse3);

fs_MAP = mean(f_pe);
mse_MAP = mean(mse4);

fs_REr = mean(f_rer);
mse_REr = mean(mse5);

fs_RMr = mean(f_rmr);
mse_RMr = mean(mse6);



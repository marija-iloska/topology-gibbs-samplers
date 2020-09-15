function [A_old] = compute_a(y, A_old, C_old, log_rho0, log_rho1, var_q, T, dx)

for j=1:dx
    for k=1:dx
        A_old(j,k) = 1;
        log_pa1 = log_rho1;
        for tau=1:T-1
            log_pa1 = log_pa1-(0.5/var_q)*(y(j,tau+1) - y(:,tau)'*(A_old(j,:).*C_old(j,:))')^2;
        end
        A_old(j,k) = 0;
        log_pa0 = log_rho0;
        for tau = 1:T-1
            log_pa0 = log_pa0-(0.5/var_q)*(y(j,tau+1) - y(:,tau)'*(A_old(j,:).*C_old(j,:))')^2;
        end
        pa0 = exp(log_pa0-max([log_pa0, log_pa1]));
        pa1 = exp(log_pa1-max([log_pa0, log_pa1]));
        alpha = pa1/(pa1+pa0);
        A_old(j,k) = rand<alpha;
    end
end
end
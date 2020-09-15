function [mu_c, sig_c] = compute_c(y, A_old, var_c, var_q,j, T, dx)

        % Determine the indicies of a_i that are nonzero
        idx=(A_old(j,:)~=0);
        
        % Determine distribution of nonzero parameters
        Cddot_prec = (1/var_q)*y(idx,1:T-1)*y(idx,1:T-1)';
        Cddot_mu = (1/var_q)*(Cddot_prec\y(idx,1:T-1)*y(j,2:T)');
        
        % Get prior components that correspond to those parameters
        Cddot_prec_prior = (1/var_c)*eye(sum(idx));
        Cddot_mu_prior = zeros(sum(idx),1);
        
        % Combine the nonzero parts with the prior
        Cddot_sig_post = inv(Cddot_prec+Cddot_prec_prior);
        Cddot_mu_post = Cddot_sig_post* ...
            (Cddot_prec_prior*Cddot_mu_prior+ Cddot_prec*Cddot_mu);
        
        % Obtain posterior mean and covariance of C
        mu_c = zeros(dx,1);          mu_c(idx) = Cddot_mu_post;
        sig_c = (1/var_c)*eye(dx);   sig_c(idx,idx)=Cddot_sig_post;
        sig_c = (sig_c+sig_c')/2;
end
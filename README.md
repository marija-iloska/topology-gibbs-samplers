# Graphical Network and Topology Estimation for Autoregressive Models using Gibbs Sampling

## Gibbs Sampling
A Gibbs sampler is a Markov chain Monte Carlo (MCMC) method which iteratively samples a parameter from its posterior conditional on the last obtained samples of the rest of the parameters in the model.
Here, we propose several strategies based on Gibbs sampling to estimate the network and topology of a 1st order VAR model.

Specifically, we developed the following Gibbs-based algorithms:
1. Element Gibbs Sampler (and reversed)
2. Matrix Gibbs Sampler (and reversed)
3. Element Blocked Gibbs Sampler (and reversed)
4. Point Estimate Gibbs samplers (Maximum Likelihood and Maximum A Posteriori)

The element samplers refer to sampling the network and topologies element by element, and respectively the matrix ones as a whole.
The term "reversed" refers to the order of sampling of the topology and the network, which in some cases is not equivalent.

### Synthetic Data
Thorough experiments based on priors, data size, and system size, as well as comparing to LASSO - a standard regularization approach are carried out.

### Real Data - Financial network
We applied the best Gibbs sampler and LASSO to data of 27 influential stock market indices on a time-series of about 30 data points, and 20 were reserved for prediction. 

For details, please refer to the following link.
#### Reference paper: http://tinyurl.com/sciencedirect-iloska-gibbs

### Code

1. Sampling.m  - the main script which generates data to user settings and runs all the samplers.
2. generate_mat.m - generates VAR data according to user settings.
3. adj_eval.m - computes F-score of the topology estimate
4. RE_gibbs.m - Element Gibbs Sampler
5. REr_gibbs.m - Element Gibbs Sampler reversed
6. RM_gibbs.m - Matrix Gibbs Sampler
7. RMr_gibbs.m - Matrix Gibbs Sampler reversed
8. JM_gibbs.m - Element Blocked Gibbs Sampler
9. JE_gibbs.m - Element Blocked Gibbs Sampler reversed
10. PE_gibbs.m - Point Estimate Gibbs Samplers (choice of ML or MAP)
    

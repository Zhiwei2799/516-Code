def generate_returns_paths(r, P, K, cov, Lambda, Delta, Rf, M):
    returns = np.zeros((M,P,K))

    for k in range(K):
        R = np.zeros((M, P))
        
        for m in range(M):
            epsilon = np.random.multivariate_normal(mean=np.zeros(P), cov=np.eye(P))
            R[m, :] = np.exp((r * np.ones(P) + cov @ Lambda - (1/2) * np.diag(cov @ cov.T)) * Delta +
                np.sqrt(Delta) * cov @ epsilon) - Rf * np.ones(P)
        returns[:,:,k] = R
        ccum_returns = np.cumprod(returns + 1, axis=2) - 1
    return returns, ccum_returns
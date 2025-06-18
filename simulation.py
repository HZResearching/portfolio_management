import numpy as np
from shared_rng import get_rng
from portfolio_construction import BaseOptimizer, CustomizedOptimizer

class SimulateT():
    def __init__(self, expected_r, cov, w, risk_avs, var_lv, cvar_pnlt, conct_pnlt, drawdn_pnlt, vol_cap, bounds, group_constraints):
        self.rng = get_rng()
        z = self.rng.standard_t(df=20, size=(100, len(expected_r)))
        chol = np.linalg.cholesky(cov)
        self.returns = expected_r + z @ chol.T
        self.current_w = w
        self.lmda = risk_avs
        self.var_lv = var_lv
        self.cvar_pnlt = cvar_pnlt
        self.conct_pnlt = conct_pnlt
        self.drawdn_pnlt = drawdn_pnlt
        self.vol_cap = vol_cap
        self.bounds = bounds
        self.group_constraints = group_constraints

    def resample_optimize(self, n_resamples):

        resample_weights = []
        for _ in range(n_resamples):
            sample_indices = self.rng.choice(len(self.returns), size=len(self.returns), replace=True)
            sample = self.returns[sample_indices]
            sample_mean = sample.mean(axis=0)
            sample_cov = np.cov(sample, rowvar=False)
        
            optimizer = CustomizedOptimizer(sample_mean, sample_cov, self.current_w, self.lmda, sample, self.var_lv, self.cvar_pnlt, self.conct_pnlt, self.drawdn_pnlt)
            res = optimizer.optimize_weights(self.group_constraints, self.bounds, self.vol_cap)
        
            if res.success:
                resample_weights.append(res.x)
    
        resampled_mvo_weights = np.mean(resample_weights, axis=0)
        return resampled_mvo_weights

class SimulateNormal():
    def __init__(self):
        self.rng = get_rng()
        
    def simulate_drawdown_probability(self, weights, mean, cov, years=10, periods_per_year=12, simulations=100, threshold=0.15):
        weights = np.asarray(weights)
        n_periods = years * periods_per_year
        mu = np.dot(mean, weights)
        sigma = np.sqrt(weights.T @ cov @ weights)
        simulated_returns = self.rng.normal(loc=mu / periods_per_year, scale=sigma / np.sqrt(periods_per_year), size=(simulations, n_periods))
        cum_returns = (1 + simulated_returns).cumprod(axis=1)
        max_drawdowns = np.max(np.maximum.accumulate(cum_returns, axis=1) - cum_returns, axis=1) / np.maximum.accumulate(cum_returns, axis=1).max(axis=1)
        prob = np.mean(max_drawdowns > threshold)
        return prob
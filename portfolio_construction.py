import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Optional

class BaseOptimizer():
    def __init__(self, returns, cov, initial_weights):
        self.returns = returns
        self.cov = cov
        self.current_weights = initial_weights

    def optimize_weights():
        raise NotImplementedError("Subclasses should implement this method.")

    def objective(self, weights, cov_matrix, returns, alpha):
        raise NotImplementedError("Subclasses should implement this method.")

    def constraints(self, group_constraints: Optional[List[Tuple[List[int], float, float]]]=None, vol_cap:Optional[float]=None, bounds:Optional[List[Tuple[float, float]]]=None):
        bounds = [(0, 1)] * len(self.current_weights) if bounds is None else bounds

        constraints = [
            {'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1)}
        ]
        if vol_cap is not None:
            constraints.append({'type': 'ineq', 'fun': lambda w: float(vol_cap - np.sqrt(np.dot(w, np.dot(self.cov, w))))})

        if group_constraints is not None:
            for group_indices, lower, upper in group_constraints:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, idxs=group_indices, lb=lower: np.sum(np.array(w)[idxs]) - lb
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, idxs=group_indices, ub=upper: ub - np.sum(np.array(w)[idxs])
                })

        return bounds, constraints


class SharpeOptimizer(BaseOptimizer):
    def __init__(self, returns, cov, initial_w, rf_annual):
        super().__init__(returns, cov, initial_w)
        self.rf_annual = rf_annual
        
    def objective(self, weights, cov_matrix, returns):
        port_return = np.dot(weights, self.returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(port_variance)
        if port_vol == 0:
            return np.inf
        sharpe = (port_return - self.rf_annual) / port_vol
        return -sharpe

    def optimize_weights(self, bounds, group_constraints: List[Tuple[List[int], float, float]], vol_cap=1.0):
        bounds, cons = self.constraints(group_constraints=group_constraints, vol_cap=vol_cap, bounds=bounds)

        result = minimize(
            fun=self.objective,
            x0=self.current_weights,
            args=(self.cov, self.returns, self.rf_annual), 
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        if not result.success:
            raise ValueError("Optimization failed: " + result.message)

        return result.x


class CustomizedOptimizer(BaseOptimizer):
    def __init__(self, mean_returns, cov, w, risk_aversion, returns, var_level, cvar_penalty, concentration_penalty, drawdown_penalty):
        super().__init__(returns, cov, w)
        self.mean_returns = mean_returns
        self.risk_aversion = risk_aversion
        self.var_level = var_level
        self.cvar_penalty = cvar_penalty
        self.concentration_penalty = concentration_penalty
        self.drawdown_penalty = drawdown_penalty

    def optimize_weights(self, group_constraints: List[Tuple[List[int], float, float]], bounds, vol_cap=1.0):
        bounds, cons = self.constraints(group_constraints=group_constraints, vol_cap=vol_cap, bounds=bounds)

        args = ((self.mean_returns,) if self.mean_returns is not None else ()) + (self.cov, self.risk_aversion, self.returns, self.var_level, self.cvar_penalty, self.concentration_penalty, self.drawdown_penalty)
        
        res = minimize(
            fun=self.objective, 
            x0=self.current_weights,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        return res
      
    def objective(self, weights, mean_returns, cov_matrix, risk_aversion, returns, alpha, cvar_penalty, concentration_penalty, drawdown_penalty):
        port_return = np.dot(weights, mean_returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        cvar = self.compute_cvar(weights, returns, alpha)
        concentration = self.risk_concentration(weights, cov_matrix)
        from simulation import SimulateNormal
        simulator = SimulateNormal()
        drawdown_prob = simulator.simulate_drawdown_probability(weights, mean_returns, cov_matrix)
        return -(port_return - 0.5 * risk_aversion * port_variance - cvar_penalty * cvar - concentration_penalty * concentration - drawdown_penalty * drawdown_prob)

    def compute_cvar(self, weights, returns, alpha=0.05):
        portfolio_returns = returns @ weights
        var_threshold = np.percentile(portfolio_returns, 100 * alpha)
        losses = -portfolio_returns[portfolio_returns <= var_threshold]
        return losses.mean() if len(losses) > 0 else 0.0

    def risk_concentration(self, weights, cov_matrix):
        port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        mrc = cov_matrix @ weights
        rc = weights * mrc / port_vol
        return np.sum((rc - rc.mean()) ** 2)

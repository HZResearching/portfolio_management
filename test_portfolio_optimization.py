
import unittest
import numpy as np
from simulation import SimulateT
from portfolio_construction import CustomizedOptimizer

class TestPortfolioOptimization(unittest.TestCase):

    def setUp(self):
        # Minimal synthetic setup for testing
        self.expected_returns = np.array([0.04, 0.05, 0.06])
        self.cov_matrix = np.array([
            [0.01, 0.002, 0.001],
            [0.002, 0.015, 0.003],
            [0.001, 0.003, 0.02]
        ])
        self.current_weights = [0.3, 0.4, 0.3]
        self.risk_aversion = 2.0
        self.alpha = 0.05
        self.cvar_penalty = 0.01
        self.concentration_penalty = 1.0
        self.drawdown_penalty = 1e-8
        self.vol_cap = 0.25
        self.bounds = [(0, 1)] * 3
        self.group_constraints = [([0, 1], 0.2, 0.9), ([2], 0.1, 1.0)]
        self.n_resamples = 5

    def test_simulate_t_resample_optimize(self):
        sim = SimulateT(
            expected_r=self.expected_returns,
            cov=self.cov_matrix,
            w=self.current_weights,
            risk_avs=self.risk_aversion,
            var_lv=self.alpha,
            cvar_pnlt=self.cvar_penalty,
            conct_pnlt=self.concentration_penalty,
            drawdn_pnlt=self.drawdown_penalty,
            vol_cap=self.vol_cap,
            bounds=self.bounds,
            group_constraints=self.group_constraints
        )
        weights = sim.resample_optimize(self.n_resamples)
        self.assertEqual(len(weights), 3)
        self.assertTrue(np.isclose(np.sum(weights), 1, atol=1e-2))

    def test_optimizer_objective_runs(self):
        returns = np.random.normal(0.05, 0.01, size=(100, 3))
        optimizer = CustomizedOptimizer(
            mean_returns=self.expected_returns,
            cov=self.cov_matrix,
            w=self.current_weights,
            risk_aversion=self.risk_aversion,
            returns=returns,
            var_level=self.alpha,
            cvar_penalty=self.cvar_penalty,
            concentration_penalty=self.concentration_penalty,
            drawdown_penalty=self.drawdown_penalty
        )
        res = optimizer.optimize_weights(group_constraints=self.group_constraints, bounds=self.bounds, vol_cap=self.vol_cap)
        self.assertTrue(res.success)
        self.assertEqual(len(res.x), 3)

if __name__ == '__main__':
    unittest.main()

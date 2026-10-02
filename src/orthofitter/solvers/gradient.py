import numpy as np
from scipy.optimize import minimize
from .base import BaseSolver

class GradientSolver(BaseSolver):
    """SciPyのL-BFGS-Bなどの勾配法を用いたローカル最適化ソルバー"""
    
    def __init__(self, catalog, environment, options=None):
        self.catalog = catalog
        self.environment = environment
        self.options = options or {'maxiter': 500, 'ftol': 1e-9}

    def optimize(self, initial_coeffs: np.ndarray) -> dict:
        def objective(coeffs):
            u_pred = self.catalog.evaluate_all(self.environment.x, coeffs)
            return self.environment.evaluate_rayleigh_loss(u_pred)

        result = minimize(
            objective,
            initial_coeffs,
            method='L-BFGS-B',
            options=self.options
        )

        return {
            'success': result.success,
            'message': result.message,
            'optimized_coeffs': result.x,
            'final_loss': result.fun
        }
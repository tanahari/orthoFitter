import numpy as np
from .base import BaseSolver
from .stochastic import StochasticSearchSolver
from .gradient import GradientSolver

class HybridSolver(BaseSolver):
    """
    確率的探索（大域的）と勾配法（局所的・仕上げ）を組み合わせたハイブリッドソルバー
    """
    
    def __init__(self, catalog, environment, options=None):
        self.catalog = catalog
        self.environment = environment
        
        # オプションのデフォルト値や個別設定
        self.options = options or {}
        
        # 内部でそれぞれのソルバーを初期化
        self.stochastic_solver = StochasticSearchSolver(
            catalog, environment, options=self.options.get('stochastic', None)
        )
        self.gradient_solver = GradientSolver(
            catalog, environment, options=self.options.get('gradient', None)
        )

    def optimize(self, initial_coeffs: np.ndarray) -> dict:
        """
        1. 確率的探索で大域的な最適解のエリア（谷）を見つける
        2. その係数を初期値として、勾配法（L-BFGS-B）で最深部まで追い込む
        """
        # --- ステージ1: 大域的探索 (Stochastic Search) ---
        stochastic_result = self.stochastic_solver.optimize(initial_coeffs)
        
        # --- ステージ2: 局所的最適化・仕上げ (Gradient Descent) ---
        intermediate_coeffs = stochastic_result['optimized_coeffs']
        gradient_result = self.gradient_solver.optimize(intermediate_coeffs)

        # 最終的な結果を統合して返す（診断用グラフ用の履歴も引き継ぐ）
        return {
            'success': gradient_result['success'],
            'message': f"Hybrid optimization converged. Final Loss: {gradient_result['final_loss']:.4e}",
            'optimized_coeffs': gradient_result['optimized_coeffs'],
            'final_loss': gradient_result['final_loss'],
            'history_loss': stochastic_result.get('history_loss', []),
            'history_sigma': stochastic_result.get('history_sigma', [])
        }
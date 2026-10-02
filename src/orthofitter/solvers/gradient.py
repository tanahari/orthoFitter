import numpy as np
from scipy.optimize import minimize
from ..catalog import HermiteCatalog
from ..environment import HermiteODEEnvironment

class CoefficientOptimizer:
    """
    「0.0001の世界」（係数の微調整）を攻略する最適化エンジン
    
    指定された次数のカタログ基底を使い、微分方程式の残差（Loss）が
    最小になるような展開係数 c_n を数値最適化で求める。
    """
    def __init__(self, catalog: HermiteCatalog, environment: HermiteODEEnvironment):
        self.catalog = catalog
        self.environment = environment

    def optimize(self, initial_coeffs: np.ndarray) -> dict:
        """
        初期係数を受け取り、残差を最小化する係数を探索する
        """
        def objective(coeffs):
            # 1. 係数から予測波形 u_pred を生成
            u_pred = self.catalog.evaluate_all(self.environment.x, coeffs)

            # 2. レイリー商ベースのロス（自明な解を排除する評価）を受け取る
            loss = self.environment.evaluate_rayleigh_loss(u_pred)
            return loss

        # SciPy の L-BFGS-B などの勾配ベース・準ニュートン法ソルバーを使用
        # (自明な解 c=0 を避けるための正規化やスケーリングを後々入れることも可能)
        result = minimize(
            objective,
            initial_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-9}
        )

        return {
            'success': result.success,
            'message': result.message,
            'optimized_coeffs': result.x,
            'final_loss': result.fun
        }
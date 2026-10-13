import numpy as np

class PeriodicODEEnvironment:
    """
    周期境界条件を持つ微分方程式の環境（ジャッジメント・残差計算器）
    
    対象とする方程式（例: プレーンなヘルムホルツ方程式 / 1次元波動方程式）:
        u''(x) + lambda_val * u(x) = 0
    """
    def __init__(self, x_domain: tuple[float, float] = (0.0, 10.0), num_points: int = 1000, lambda_val: float = 1.0):
        self.x_min, self.x_max = x_domain
        self.num_points = num_points
        self.lambda_val = lambda_val

        # 評価用の格子点 x
        self.x = np.linspace(self.x_min, self.x_max, self.num_points)
        self.dx = self.x[1] - self.x[0]

        # 数値微分の端点アーティファクトを除外するためのトリミング幅
        self._trim = 2

    def compute_residuals(self, u_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        微分方程式の内部残差 R(x) を計算する
        R(x) = u''(x) + lambda_val * u(x)
        """
        du_dx = np.gradient(u_pred, self.dx)
        d2u_dx2 = np.gradient(du_dx, self.dx)

        residuals_full = d2u_dx2 + self.lambda_val * u_pred

        # エルミート系と同様に、微分誤差が大きくなる両端の数点をスライスして除外
        x_sub = self.x[self._trim:-self._trim]
        residuals = residuals_full[self._trim:-self._trim]

        return residuals, x_sub

    def evaluate_rayleigh_loss(self, u_pred: np.ndarray) -> float:
        """
        レイリー商ロス + 周期境界条件（値と1階微分の連続性）のペナルティを統合した評価関数
        """
        # 1. 内部の残差（端がトリミングされたもの）を取得
        residuals, x_sub = self.compute_residuals(u_pred)
        
        # 分母の積分用にも、同じ範囲で u_pred をスライスして対応させる
        u_sub = u_pred[self._trim:-self._trim]
        
        residual_norm = np.trapezoid(residuals**2, x_sub)
        solution_norm = np.trapezoid(u_sub**2, x_sub)
        
        # ゼロ解の罠を排除
        if solution_norm < 1e-4:
            return float('inf')
            
        base_loss = residual_norm / solution_norm

        # 2. 周期境界条件（BC）ペナルティの付加
        # 端点での値のズレ: u(x_min) vs u(x_max)
        bc_value_diff = (u_pred[0] - u_pred[-1])**2
        
        # 端点での傾き（1階微分）のズレ: u'(x_min) vs u'(x_max)
        du_dx = np.gradient(u_pred, self.dx)
        bc_grad_diff = (du_dx[0] - du_dx[-1])**2
        
        # ペナルティ係数（両端が滑らかにつながるように誘導）
        alpha_bc = 10.0 
        bc_penalty = alpha_bc * (bc_value_diff + bc_grad_diff)

        # 最終的なロス = 方程式の残差ロス + 境界条件ペナルティ
        total_loss = base_loss + bc_penalty
        
        return float(total_loss)
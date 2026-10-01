import numpy as np

class HermiteODEEnvironment:
    """
    エルミート微分方程式の環境（ジャッジメント・残差計算器）
    
    対象とする方程式:
        u''(x) - 2x * u'(x) + lambda_val * u(x) = 0
    今回は lambda_val = 2 をターゲットにする。
    """
    def __init__(self, x_domain: tuple[float, float] = (-5.0, 5.0), num_points: int = 1000, lambda_val: float = 2.0):
        """
        Parameters:
        - x_domain: 評価する空間のドメイン (x_min, x_max)
        - num_points: サンプリング点数（空間の解像度）
        - lambda_val: 微分方程式のパラメータ lambda (今回は 2.0)
        """
        self.x_min, self.x_max = x_domain
        self.num_points = num_points
        self.lambda_val = lambda_val

        # 評価用の格子点 x をあらかじめ用意しておく
        self.x = np.linspace(self.x_min, self.x_max, self.num_points)
        self.dx = self.x[1] - self.x[0] # 数値微分用の間隔

    def compute_residuals(self, u_pred: np.ndarray) -> np.ndarray:
        """
        予測関数 u_pred(x) の格子点上での微分方程式の残差 R(x) を計算する
        
        R(x) = u''(x) - 2x * u'(x) + lambda * u(x)
        """
        # 1階微分 u'(x) を中央差分などで計算 (numpy.gradient は便利)
        du_dx = np.gradient(u_pred, self.dx)

        # 2階微分 u''(x) を計算
        d2u_dx2 = np.gradient(du_dx, self.dx)

        # 微分方程式の左辺に代入して残差を算出
        residuals = d2u_dx2 - 2.0 * self.x * du_dx + self.lambda_val * u_pred

        return residuals

    def evaluate_rayleigh_loss(self, u_pred: np.ndarray) -> float:
        """
        レイリー商スタイルのロスを計算する
        Loss = (残差の二乗積分) / (解の二乗積分)
        
        これによって u(x) = 0 という自明な解（ゼロ解の罠）を完全に排除する。
        """
        residuals = self.compute_residuals(u_pred)
        
        # 分子: 残差の2乗積分
        residual_norm = np.trapezoid(residuals**2, self.x)
        
        # 分母: 解自体の2乗積分（エネルギー / ノルム）
        solution_norm = np.trapezoid(u_pred**2, self.x)
        
        # ゼロ割を防ぐための微小な定数 (eps)
        eps = 1e-8
        
        # 「残差が小さく」かつ「解がゼロではない（分母が大きい）」ほどロスが小さくなる
        loss = residual_norm / (solution_norm + eps)
        
        return float(loss)
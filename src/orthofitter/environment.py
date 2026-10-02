import numpy as np

class HermiteODEEnvironment:
    """
    量子調和振動子型微分方程式の環境（ジャッジメント・残差計算器）
    
    対象とする方程式（ガウス重み付きエルミート関数が解になる形）:
        u''(x) - x^2 * u(x) + lambda_val * u(x) = 0
    今回は n = 1 の状態をターゲットにするため、固有パラメータは lambda_val = 3.0 とする。
    """
    def __init__(self, x_domain: tuple[float, float] = (-5.0, 5.0), num_points: int = 1000, lambda_val: float = 3.0):
        """
        Parameters:
        - x_domain: 評価する空間のドメイン (x_min, x_max)
        - num_points: サンプリング点数（空間の解像度）
        - lambda_val: 微分方程式のパラメータ lambda (n=1 の理論値は 3.0)
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
        
        R(x) = u''(x) - x^2 * u(x) + lambda_val * u(x)
        """
        # 1階微分を経由して 2階微分 u''(x) を計算
        du_dx = np.gradient(u_pred, self.dx)
        d2u_dx2 = np.gradient(du_dx, self.dx)

        # 微分方程式の左辺に代入して残差を算出（ポテンシャル項 -x^2 * u を含む）
        residuals = d2u_dx2 - (self.x ** 2) * u_pred + self.lambda_val * u_pred

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
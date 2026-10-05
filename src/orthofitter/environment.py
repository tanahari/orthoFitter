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

        # private
        self._trim = 2

    def compute_residuals(self, u_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        予測関数 u_pred(x) の格子点上での微分方程式の残差 R(x) を計算する
        
        R(x) = u''(x) - x^2 * u(x) + lambda_val * u(x)
        """
        # 1. まず全区間でキレイに1階・2階微分を計算する
        du_dx = np.gradient(u_pred, self.dx)
        d2u_dx2 = np.gradient(du_dx, self.dx)

        # 2. 全区間で残差を算出する
        residuals_full = d2u_dx2 - (self.x ** 2) * u_pred + self.lambda_val * u_pred

        # 3. 微分誤差が大きくなる両端の数点を、ここで初めて一括してスライス・除外する
        x_sub = self.x[self._trim:-self._trim]
        residuals = residuals_full[self._trim:-self._trim]

        return residuals, x_sub

    def evaluate_rayleigh_loss(self, u_pred: np.ndarray) -> float:
        """
        レイリー商スタイルのロスを計算する
        Loss = (残差の二乗積分) / (解の二乗積分)
        
        これによって u(x) = 0 という自明な解（ゼロ解の罠）を完全に排除する。
        """
        # compute_residuals からすでに端が除外されたクリーンな residuals と x_sub を受け取る
        residuals, x_sub = self.compute_residuals(u_pred)

        # 分母の積分用にも、同じ範囲で u_pred をスライスする
        u_sub = u_pred[self._trim:-self._trim]
        
        # 分子: 残差の2乗積分
        residual_norm = np.trapezoid(residuals**2, x_sub)
        
        # 分相: 解自体の2乗積分
        solution_norm = np.trapezoid(u_sub**2, x_sub)
        
        # ゼロ割を防ぐための微小な定数 (eps)
        eps = 1e-8
        
        # 「残差が小さく」かつ「解がゼロではない（分母が大きい）」ほどロスが小さくなる
        loss = residual_norm / (solution_norm + eps)
        
        return float(loss)
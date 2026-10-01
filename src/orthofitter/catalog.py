import numpy as np
from scipy.special import hermite

class HermiteCatalog:
    """
    エルミート多項式をベースにした直交関数カタログ
    """
    def __init__(self, max_degree: int = 5, use_weight: bool = True):
        """
        Parameters:
        - max_degree: 考慮する最大次数（n = 0, 1, 2, ..., max_degree）
        - use_weight: ガウス型重み exp(-x^2 / 2) をかけるかどうか
                      (L2空間での直交性を保つためによく使われる)
        """
        self.max_degree = max_degree
        self.use_weight = use_weight

    def evaluate(self, n: int, x: np.ndarray) -> np.ndarray:
        """
        n次のエルミート基底関数 phi_n(x) の値を計算する
        """
        # scipy.special.hermite(n) は n 次のエルミート多項式オブジェクトを返す
        h_n = hermite(n)
        val = h_n(x)

        if self.use_weight:
            # 物理で一般的なガウス重みを付与
            val = val * np.exp(-x**2 / 2.0)

        return val

    def evaluate_all(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        係数ベクトル coeffs = [c_0, c_1, ..., c_N] を受けて、
        線形結合された予測関数 u_pred(x) = sum(c_n * phi_n(x)) を計算する
        
        これがホワイトボックスの「数式のレシピの具現化」にあたります。
        """
        u_pred = np.zeros_like(x, dtype=float)
        for n, c in enumerate(coeffs):
            if n > self.max_degree:
                break
            if c != 0.0:
                u_pred += c * self.evaluate(n, x)
        return u_pred
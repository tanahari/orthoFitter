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

class TrigonometricCatalog:
    """
    周期解や波動系をターゲットにした三角関数（Fourier系）ベースの直交関数カタログ。
    HermiteCatalog と同一のインターフェースを持ち、GAやソルバーからそのまま呼び出し可能。
    """
    def __init__(self, max_degree: int = 5, domain_length: float = 10.0):
        """
        Parameters:
        - max_degree: 考慮する最大次数（基底の総数は max_degree + 1）
        - domain_length: 空間ドメインのスケール (例: [-L/2, L/2] または周期 L)
        """
        self.max_degree = max_degree
        self.domain_length = domain_length

    def evaluate(self, n: int, x: np.ndarray) -> np.ndarray:
        """
        n次（インデックス n）に対応する三角関数基底 phi_n(x) を計算する。
        - n = 0: 定数項 (1.0)
        - nが奇数: sin(k * pi * x / L)
        - nが偶数: cos(k * pi * x / L)
        """
        if n == 0:
            return np.ones_like(x, dtype=float)
        
        # 次数から周波数 k を割り振る例（n=1->sin, n=2->cos, n=3->sin(2x), ...）
        k = (n + 1) // 2
        L = self.domain_length
        
        if n % 2 == 1:
            # 奇数番目: サイン基底
            return np.sin(2.0 * np.pi * k * x / L)
        else:
            # 偶数番目: コサイン基底
            return np.cos(2.0 * np.pi * k * x / L)

    def evaluate_all(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        係数ベクトルを受けて、線形結合された予測関数 u_pred(x) = sum(c_n * phi_n(x)) を計算する。
        """
        u_pred = np.zeros_like(x, dtype=float)
        for n, c in enumerate(coeffs):
            if n > self.max_degree:
                break
            if c != 0.0:
                u_pred += c * self.evaluate(n, x)
        return u_pred
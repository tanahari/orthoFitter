from abc import ABC, abstractmethod
import numpy as np

class BaseSolver(ABC):
    """すべての最適化・探索ソルバーの抽象基底クラス"""

    @abstractmethod
    def optimize(self, initial_coeffs: np.ndarray) -> dict:
        """
        初期係数を受け取り、最適化を実行して結果の辞書を返す。
        戻り値のフォーマットは全ソルバーで統一する。
        """
        pass
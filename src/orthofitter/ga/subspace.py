import numpy as np

class SubspaceCatalogWrapper:
    """
    【サブ空間用カタログ・ラッパー】
    HybridSolver がカタログの評価をする際に、
    部分空間の係数をフルサイズに復元して大元のカタログに渡す仲介役。
    """
    def __init__(self, original_catalog, active_indices: np.ndarray):
        self.original_catalog = original_catalog
        self.active_indices = active_indices
        self.max_degree = len(active_indices) - 1  # サブ空間上の次数扱い

    def evaluate_all(self, x, active_coeffs: np.ndarray):
        full_coeffs = np.zeros(self.original_catalog.max_degree + 1)
        full_coeffs[self.active_indices] = active_coeffs
        return self.original_catalog.evaluate_all(x, full_coeffs)


class SubspaceEnvironmentWrapper:
    """
    【サブ空間用環境・ラッパー】
    GAのマスク（0/1）で選択された有効な基底の次元だけを切り出し、
    HybridSolver が「この部分空間が全次元の世界だ」と思って動けるようにするクラス。
    """
    def __init__(self, catalog, environment, mask: np.ndarray):
        self.original_catalog = catalog
        self.original_environment = environment
        self.mask = mask
        
        # マスクが 1 のインデックス（有効な基底の次元）だけを抽出
        self.active_indices = np.where(mask == 1)[0]
        self.subspace_dim = len(self.active_indices)
        
        # HybridSolver が参照する環境のプロパティを引き継ぐ
        self.x = environment.x
        self.num_points = environment.num_points
        self.lambda_val = environment.lambda_val

    def evaluate_rayleigh_loss(self, u_pred: np.ndarray) -> float:
        """
        ソルバーから渡された予測波形 u_pred を受け取り、
        大元の環境のレイリー商ロス計算へそのまま転送する。
        （係数の復元は SubspaceCatalogWrapper 側ですでに完了しているため不要）
        """
        return self.original_environment.evaluate_rayleigh_loss(u_pred)
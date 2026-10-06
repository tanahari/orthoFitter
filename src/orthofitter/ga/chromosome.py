import numpy as np
import uuid

class Chromosome:
    """
    GAの1個体を表現するクラス。
    どの基底を使うかの構造(mask)と、その構造下での最適係数(coeffs)を保持する。
    """
    def __init__(self, max_degree: int, parent_ids: list[str] = None):
        self.max_degree = max_degree
        self.dimension = max_degree + 1

        # --- 系統樹追跡用のIDと親ID ---
        self.id = str(uuid.uuid4())[:8]  # 8桁のランダムな一意ID
        self.parent_ids = parent_ids if parent_ids is not None else []
        
        # 1/0のマスクをランダムに初期化
        self.mask = np.random.choice([0, 1], size=self.dimension)
            
        # 全基底分の係数を保持する配列
        self.coeffs = np.zeros(self.dimension)
        
        # 【修正】マスクが1の初期位置には、最初から微小なランダム値を与えておく
        for i in range(self.dimension):
            if self.mask[i] == 1:
                self.coeffs[i] = np.random.normal(0, 0.1)
                
        self.loss = float('inf')
        self.fitness = -float('inf')

    def mutate(self, mutation_rate: float):
        """
        指定された確率で 1/0 のビットを反転させる（構造のワープ）。
        """
        for i in range(self.dimension):
            if np.random.rand() < mutation_rate:
                self.mask[i] = 1 - self.mask[i] # 1なら0、0なら1へ反転
                
                # 新しくONになった基底は微小なランダム値で初期化
                if self.mask[i] == 1:
                    self.coeffs[i] = np.random.normal(0, 0.1)
                else:
                    self.coeffs[i] = 0.0
        
        # 全てOFFになるのを防ぐ安全装置
        if np.sum(self.mask) == 0:
            self.mask[np.random.randint(0, self.dimension)] = 1

    def crossover(self, other: 'Chromosome') -> 'Chromosome':
        """
        一様交叉（Uniform Crossover）で2つの親から子を生成する。
        """
        # 子を生成する際に、両親のIDを親情報として引き継ぐ
        child = Chromosome(self.max_degree, parent_ids=[self.id, other.id])
        
        for i in range(self.dimension):
            # 50%の確率で親Aか親Bのマスクを引き継ぐ
            if np.random.rand() < 0.5:
                child.mask[i] = self.mask[i]
                child.coeffs[i] = self.coeffs[i]
            else:
                child.mask[i] = other.mask[i]
                child.coeffs[i] = other.coeffs[i]
                
        if np.sum(child.mask) == 0:
            child.mask[np.random.randint(0, self.dimension)] = 1
            
        return child
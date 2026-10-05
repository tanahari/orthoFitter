import numpy as np
import copy
from orthofitter.ga.chromosome import Chromosome
from orthofitter.solvers.gradient_and_stochastic import HybridSolver
from orthofitter.ga.subspace import SubspaceCatalogWrapper, SubspaceEnvironmentWrapper

class GAOptimizer:
    def __init__(self, catalog, environment, options=None):
        self.catalog = catalog
        self.environment = environment
        self.options = options or {}
        
        self.pop_size = self.options.get('pop_size', 20)
        self.generations = self.options.get('generations', 10)
        self.mutation_rate = self.options.get('mutation_rate', 0.1)
        self.elite_ratio = self.options.get('elite_ratio', 0.1)
        self.beta = self.options.get('beta', 1e-3) # スパース化のペナルティ係数
        
        self.solver_options = self.options.get('solver_options', {})
        self.num_elites = max(1, int(self.pop_size * self.elite_ratio))
        self.max_degree = self.catalog.max_degree

    def _evaluate_fitness(self, ind: Chromosome):
        active_indices = np.where(ind.mask == 1)[0]
        
        if len(active_indices) == 0:
            ind.loss = float('inf')
            ind.fitness = -float('inf')
            return

        active_initial_coeffs = ind.coeffs[active_indices]

        # subspace.py のラッパーを使って綺麗にカプセル化
        proxy_env = SubspaceEnvironmentWrapper(self.catalog, self.environment, ind.mask)
        proxy_cat = SubspaceCatalogWrapper(self.catalog, active_indices)

        # 自作の HybridSolver を実行
        solver = HybridSolver(proxy_cat, proxy_env, options=self.solver_options)
        result = solver.optimize(active_initial_coeffs)

        ind.loss = result['final_loss']
        ind.coeffs = np.zeros(self.max_degree + 1)
        ind.coeffs[active_indices] = result['optimized_coeffs']
        
        num_active = len(active_indices)
        if np.isinf(ind.loss) or np.isnan(ind.loss):
            ind.fitness = -float('inf')
        else:
            ind.fitness = 1.0 / (ind.loss + 1e-8) * (1.0 / (self.beta * num_active + 1))

    def optimize(self):
        population = [Chromosome(self.max_degree) for _ in range(self.pop_size)]
        best_overall = None

        for gen in range(self.generations):
            # ▼ 個体ごとの進捗を表示するために enumerate を使用
            for i, ind in enumerate(population):
                print(f"Generation {gen+1}/{self.generations} | Evaluating Individual {i+1}/{self.pop_size}...", end="\r")
                self._evaluate_fitness(ind)
            
            # 改行して世代ごとの結果を表示
            print(" " * 80, end="\r") # 前の行の表示をクリア（必要に応じて）
                
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # ▼ ここから：全個体の詳細を出力
            print(f"\n--- [Generation {gen+1}] 全個体のリスト（適応度順） ---")
            for i, ind in enumerate(population):
                active_indices = np.where(ind.mask == 1)[0].tolist()
                print(f"  個体 {i+1:2d} | Fitness: {ind.fitness:10.4f} | Loss: {ind.loss:10.4e} | Active数: {len(active_indices):2d} | 基底: {active_indices}")
                # 係数（coeffs）も詳しく見たければ、下の行のコメントアウトを外してください
                print(f"         Coeffs: {ind.coeffs[ind.mask == 1]}")
            print("-" * 60)
            # ▲ ここまで

            best_gen = population[0]
            print(f"Generation {gen+1} 最良個体 -> Loss: {best_gen.loss:.4e} | Active Bases: {np.sum(best_gen.mask)} | Fitness: {best_gen.fitness:.4f}\n")
            
            if best_overall is None or best_gen.fitness > best_overall.fitness:
                best_overall = copy.deepcopy(best_gen)

            new_population = []
            for i in range(self.num_elites):
                new_population.append(copy.deepcopy(population[i]))
                
            while len(new_population) < self.pop_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                child = parent1.crossover(parent2)
                child.mutate(self.mutation_rate)
                new_population.append(child)
                
            population = new_population

        return {
            'success': True,
            'best_mask': best_overall.mask.astype(int).tolist(),
            'optimized_coeffs': best_overall.coeffs,
            'final_loss': best_overall.loss,
            'best_fitness': best_overall.fitness
        }
        
    def _tournament_select(self, population: list, k: int = 3) -> Chromosome:
        candidates = np.random.choice(population, size=k, replace=False)
        return max(candidates, key=lambda x: x.fitness)
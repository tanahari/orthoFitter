import numpy as np
from .base import BaseSolver

class StochasticSearchSolver(BaseSolver):
    """
    クロスエントロピー法ベースの確率的大域的探索ソルバー
    （「パラメータの荒野」を攻略するための強化学習ミニマム実装）
    """
    
    def __init__(self, catalog, environment, options=None):
        self.catalog = catalog
        self.environment = environment
        self.options = options or {
            'maxiter': 100,
            'num_agents': 100,
            'top_k': 50,
            'initial_sigma': 1.0,
            'sigma_min': 0.01
        }

    def optimize(self, initial_coeffs: np.ndarray) -> dict:
        dimension = len(initial_coeffs)
        mu = initial_coeffs.copy()
        sigma = self.options['initial_sigma']
        sigma_min = self.options['sigma_min']

        maxiter = self.options['maxiter']
        num_agents = self.options['num_agents']
        top_k = self.options['top_k']

        best_loss = float('inf')
        best_coeffs = mu.copy()

        # --- 診断用の履歴記録リスト ---
        history_loss = []
        history_sigma = []

        for gen in range(maxiter):
            # 1. 行動（Action）：現在の中心 mu と広がり sigma から候補をガウスサンプリング
            actions = np.random.normal(loc=mu, scale=sigma, size=(num_agents, dimension))
            
            rewards = np.zeros(num_agents)
            for i in range(num_agents):
                coeffs = actions[i]
                
                # 2. 評価（Environment）：レイリー商ベースの損失を計算
                u_pred = self.catalog.evaluate_all(self.environment.x, coeffs)
                loss = self.environment.evaluate_rayleigh_loss(u_pred)
                
                # Loss の最小化を、報酬の最大化（-Loss）に変換
                rewards[i] = -loss

                # ベスト記録の更新
                if loss < best_loss:
                    best_loss = loss
                    best_coeffs = coeffs.copy()

            # 3. 学習（Update）：報酬が高い上位のエリートを抽出し、重心 mu を更新
            elite_indices = np.argsort(rewards)[-top_k:]
            elite_actions = actions[elite_indices]
            
            mu = np.mean(elite_actions, axis=0)
            
            # 分散の更新（縮小しすぎないように下限を設ける）
            current_std = np.std(elite_actions, axis=0).mean()
            sigma = max(current_std * 1.1, sigma_min)

            # --- 履歴を保存 ---
            history_loss.append(best_loss)
            history_sigma.append(sigma)

        return {
            'success': True,
            'message': f"Stochastic search converged with Best Loss: {best_loss:.4e}",
            'optimized_coeffs': best_coeffs,
            'final_loss': best_loss,
            'history_loss': history_loss,    # 追加
            'history_sigma': history_sigma   # 追加
        }
import argparse
import numpy as np

from orthofitter.catalog import HermiteCatalog
from orthofitter.environment import HermiteODEEnvironment
from orthofitter.ga.optimizer import GAOptimizer
from orthofitter.visualization import save_optimization_plots

def main():
    parser = argparse.ArgumentParser(description="OrthoFitter GA Optimization Example")
    parser.add_argument('-v', '--visualize', action='store_true', default=False)
    args = parser.parse_args()

    # n=1 の状態を狙うため lambda = 3.0
    lambda_val = 3.0
    print(f"=== OrthoFitter GA: 02_simple_ode.py (λ = {lambda_val}) ===")

    # 1. カタログの初期化（探索空間を広げるため max_degree=10）
    max_degree = 10
    catalog = HermiteCatalog(max_degree=max_degree)

    # 2. 環境の初期化
    environment = HermiteODEEnvironment(x_domain=(-5.0, 5.0), num_points=1000, lambda_val=lambda_val)

    # 3. GAオプティマイザーの初期化
    # betaはペナルティ係数。この値の調整がスパース化の鍵になります。
    ga_options = {
        'pop_size': 20,          # 1世代あたりの個体数
        'generations': 10,       # 世代数
        'mutation_rate': 0.1,    # 0/1反転の確率
        'elite_ratio': 0.1,      # 次世代へ無条件で残すエリートの割合
        'beta': 1e-3             # 基底数に対するペナルティ係数
    }
    optimizer = GAOptimizer(catalog, environment, options=ga_options)

    # 4. 最適化を実行（初期値はGAが内部でランダム生成するため不要）
    print("GAによる構造探索 & 連続最適化を実行中...")
    result = optimizer.optimize()

    print("\n--- 最終結果 ---")
    print(f"成功判定: {result['success']}")
    print(f"発見された最適マスク (1/0): {result['best_mask']}")
    print(f"最適化された係数レシピ: {result['optimized_coeffs']}")
    print(f"最終的な残差ロス (Loss): {result['final_loss']:.2e}")
    print(f"最終的な適応度 (Fitness): {result['best_fitness']:.4f}")

    if args.visualize:
        print("\n可視化モードが有効です。画像を作成しています...")
        title = f"GA OrthoFitter: Hermite ODE ($\\lambda$ = {lambda_val})"
        save_optimization_plots(environment, catalog, result, output_dir="outputs", title=title)

if __name__ == '__main__':
    main()
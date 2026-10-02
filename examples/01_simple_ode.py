import argparse
import numpy as np

# 作成したパッケージから必要なモジュールをインポート
from orthofitter.catalog import HermiteCatalog
from orthofitter.environment import HermiteODEEnvironment
from orthofitter.solvers import StochasticSearchSolver as Solver
from orthofitter.visualization import save_optimization_plots  # 共通化された可視化をインポート

def main():
    # --- コマンドライン引数の設定 ---
    parser = argparse.ArgumentParser(description="OrthoFitter ODE Optimization Example")
    parser.add_argument(
        '-v', '--visualize', 
        action='store_true', 
        default=False, 
        help='最適化の過程や結果を可視化して outputs/ に画像を出力する (デフォルト: False)'
    )
    args = parser.parse_args()

    print("=== OrthoFitter: 01_simple_ode.py (λ = 2 Hermite ODE Test) ===")

    # 1. カタログの初期化
    max_degree = 3
    catalog = HermiteCatalog(max_degree=max_degree)

    # 2. 環境の初期化
    environment = HermiteODEEnvironment(x_domain=(-5.0, 5.0), num_points=1000, lambda_val=2.0)

    # 3. ソルバーの初期化
    optimizer = Solver(catalog, environment)

    # 4. 初期係数の設定
    initial_coeffs = np.array([0.1, 1.0, 0.1, 0.1])
    print(f"初期係数レシピ: {initial_coeffs}")

    # 5. 最適化を実行
    print("最適化を実行中...")
    result = optimizer.optimize(initial_coeffs)

    print("\n--- 最終結果 ---")
    print(f"成功判定: {result['success']}")
    print(f"メッセージ: {result['message']}")
    print(f"最適化された係数レシピ: {result['optimized_coeffs']}")
    print(f"最終的な残差ロス (Loss): {result['final_loss']:.2e}")

    # 6. 可視化フラグが True の場合のみ、共通関数を実行
    if args.visualize:
        print("\n可視化モードが有効です。画像を作成しています...")
        save_optimization_plots(environment, catalog, result, output_dir="outputs")
    else:
        print("\n(※ 可視化はスキップされました。有効にするには `-v` または `--visualize` オプションを付与してください)")

if __name__ == '__main__':
    main()
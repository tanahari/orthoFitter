import argparse
import sys
from pathlib import Path

from orthofitter.catalog import HermiteCatalog
from orthofitter.environment import HermiteODEEnvironment
from orthofitter.ga.optimizer import GAOptimizer
from orthofitter.visualization import save_optimization_plots, save_basis_adoption_heatmap, save_loss_distribution_plot, save_diagnostic_dashboard

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
        'elite_ratio': 0.3,      # 次世代へ無条件で残すエリートの割合
        'beta': 25.0              # 基底数に対するペナルティ係数
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

        # 1. 従来のODE結果・診断グラフ
        save_optimization_plots(environment, catalog, result, output_dir="outputs", title=title)
        
        # 2. 基底の採用率時系列ヒートマップ
        save_basis_adoption_heatmap(result, output_dir="outputs")

        # 3. Loss分布の推移グラフ
        save_loss_distribution_plot(result, output_dir="outputs")

        # 4. 【新規追加】4大診断軸の統合ダッシュボード
        save_diagnostic_dashboard(result, output_dir="outputs")

class Tee:
    """標準出力とファイルを同時に書き込むためのクラス"""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # すぐにバッファを書き込ませるために必要

    def flush(self):
        for f in self.files:
            f.flush()

if __name__ == '__main__':
    # 1. 出力先ディレクトリの作成とファイルオープン
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "output.log"
    
    log_file = open(log_path, "w", encoding="utf-8")
    
    # 2. ターミナル（元のsys.stdout）とログファイルの両方に流すTeeを設定
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    
    try:
        # 3. メイン関数の実行
        main()
    finally:
        # 4. 確実にファイルを閉じ、標準出力を元に戻す
        log_file.close()
        sys.stdout = original_stdout
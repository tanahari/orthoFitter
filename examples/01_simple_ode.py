import numpy as np
import matplotlib.pyplot as plt

# 作成したパッケージからモジュールをインポート
from orthofitter.catalog import HermiteCatalog
from orthofitter.environment import HermiteODEEnvironment
from orthofitter.solver import CoefficientOptimizer

def main():
    print("=== OrthoFitter: 01_simple_ode.py (λ = 2 Hermite ODE Test) ===")

    # 1. カタログの初期化 (最大次数 3: n = 0, 1, 2, 3 を考慮)
    max_degree = 3
    catalog = HermiteCatalog(max_degree=max_degree)

    # 2. 環境の初期化 (ドメイン [-5, 5], λ = 2.0)
    environment = HermiteODEEnvironment(x_domain=(-5.0, 5.0), num_points=1000, lambda_val=2.0)

    # 3. ソルバーの初期化
    optimizer = CoefficientOptimizer(catalog=catalog, environment=environment)

    # 4. 初期係数の設定 (あえて少しランダム、あるいは偏った初期値を与える)
    # 例: [c_0, c_1, c_2, c_3] = [0.5, 0.5, 0.5, 0.5]
    initial_coeffs = np.array([0.5, 0.5, 0.5, 0.5])
    print(f"初期係数レシピ: {initial_coeffs}")

    # 5. 最適化（「0.0001の世界」の攻略）を実行！
    print("最適化を実行中...")
    result = optimizer.optimize(initial_coeffs)

    print("\n--- 最期の結果 ---")
    print(f"成功判定: {result['success']}")
    print(f"メッセージ: {result['message']}")
    print(f"最適化された係数レシピ: {result['optimized_coeffs']}")
    print(f"最終的な残差ロス (Loss): {result['final_loss']:.2e}")

    # 6. 結果の可視化 (Matplotlib)
    x = environment.x
    u_optimized = catalog.evaluate_all(x, result['optimized_coeffs'])
    residuals = environment.compute_residuals(u_optimized)

    # 上段：求まった近似解の波形
    plt.subplot(2, 1, 1)
    plt.plot(x, u_optimized, label=f'OrthoFitter Solution (Loss={result["final_loss"]:.2e})', color='blue', linewidth=2)
    plt.title('OrthoFitter: Hermite ODE ($\lambda = 2$) Result')
    plt.xlabel('x')
    plt.ylabel('u_pred(x)')
    plt.grid(True)
    plt.legend()

    # 下段：各点での残差 R(x) の分布
    plt.subplot(2, 1, 2)
    plt.plot(x, residuals, label='Residual $R(x)$', color='red', linestyle='--')
    plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
    plt.title('Differential Equation Residuals')
    plt.xlabel('x')
    plt.ylabel('R(x)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('result_ode.png', dpi=300)
    print("\n可視化結果を 'result_ode.png' として保存しました！")
    plt.show()

if __name__ == '__main__':
    main()
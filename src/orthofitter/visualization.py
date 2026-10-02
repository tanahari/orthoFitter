from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def save_optimization_plots(
    environment, 
    catalog, 
    result: dict, 
    output_dir: str | Path = "outputs"
):
    """
    最適化結果および診断用グラフを指定ディレクトリに保存する共通関数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = environment.x
    optimized_coeffs = result['optimized_coeffs']
    u_optimized = catalog.evaluate_all(x, optimized_coeffs)
    residuals = environment.compute_residuals(u_optimized)

    # --- 1. 解の波形と残差のプロット ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    
    axes[0].plot(x, u_optimized, label=f'OrthoFitter Solution (Loss={result["final_loss"]:.2e})', color='blue', linewidth=2)
    axes[0].set_title(r'OrthoFitter: Hermite ODE ($\lambda = 2$) Result')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u_pred(x)')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x, residuals, label='Residual $R(x)$', color='red', linestyle='--')
    axes[1].axhline(0, color='black', linewidth=0.5, linestyle=':')
    axes[1].set_title('Differential Equation Residuals')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('R(x)')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    path_ode = output_dir / 'result_ode.png'
    plt.savefig(path_ode, dpi=300)
    plt.close()
    print(f"-> 成果物を保存しました: {path_ode}")

    # --- 2. 探索の診断用プロット (Loss と Sigma の推移) ---
    if 'history_loss' in result and 'history_sigma' in result:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        axes[0].plot(result['history_loss'], color='tab:blue', linewidth=1.5)
        axes[0].set_yscale('log')
        axes[0].set_title('Best Loss Convergence')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Loss (log scale)')
        axes[0].grid(True)
        
        axes[1].plot(result['history_sigma'], color='tab:orange', linewidth=1.5)
        axes[1].set_title('Search Radius ($\sigma$)')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Sigma')
        axes[1].grid(True)

        plt.tight_layout()
        path_diag = output_dir / 'optimization_diagnostics.png'
        plt.savefig(path_diag, dpi=300)
        plt.close()
        print(f"-> 診断用グラフを保存しました: {path_diag}")
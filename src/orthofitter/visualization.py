from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def save_optimization_plots(
    environment, 
    catalog, 
    result: dict, 
    output_dir: str | Path = "outputs",
    title: str = "Result"
):
    """
    最適化結果および診断用グラフを指定ディレクトリに保存する共通関数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = environment.x
    optimized_coeffs = result['optimized_coeffs']
    u_optimized = catalog.evaluate_all(x, optimized_coeffs)
    residuals, x_sub = environment.compute_residuals(u_optimized)

    # --- 1. 解の波形と残差のプロット ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    
    axes[0].plot(x, u_optimized, label=f'OrthoFitter Solution (Loss={result["final_loss"]:.2e})', color='blue', linewidth=2)
    axes[0].set_title("OrthoFitte:" + title)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u_pred(x)')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x_sub, residuals, label='Residual $R(x)$', color='red', linestyle='--')
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
        axes[1].set_title('Search Radius ($\\sigma$)')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Sigma')
        axes[1].grid(True)

        plt.tight_layout()
        path_diag = output_dir / 'optimization_diagnostics.png'
        plt.savefig(path_diag, dpi=300)
        plt.close()
        print(f"-> 診断用グラフを保存しました: {path_diag}")

def save_coefficient_dynamics_plots(
    environment, 
    catalog, 
    result: dict, 
    output_dir: str | Path = "outputs"
):
    """
    係数のダイナミクス（世代ごとの推移）と波形の収束過程を可視化して保存する
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history_coeffs = result.get('history_coeffs', [])
    if len(history_coeffs) == 0:
        print("-> 警告: history_coeffs が見つからないため、係数ダイナミクスの可視化をスキップします。")
        return

    history_coeffs = np.array(history_coeffs)
    x = environment.x
    num_gens, dim = history_coeffs.shape

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 1. 係数の推移グラフ (アイデア3) ---
    for i in range(dim):
        axes[0].plot(history_coeffs[:, i], label=f'c_{i}', linewidth=2)
    axes[0].set_title('Coefficient Evolution over Generations')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Coefficient Value')
    axes[0].grid(True)
    axes[0].legend()

    # --- 2. 波形の重ね合わせプロット (アイデア1) ---
    # 世代の進行に合わせて、いくつかのスナップショットを抽出して色を変えて重ねる
    indices = np.linspace(0, num_gens - 1, min(10, num_gens), dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    
    for idx, color in zip(indices, colors):
        c = history_coeffs[idx]
        u_step = catalog.evaluate_all(x, c)
        axes[1].plot(x, u_step, color=color, alpha=0.8, linewidth=1.5, label=f'Gen {idx}')
        
    # 最後の最適化結果も強調して重ねる
    u_final = catalog.evaluate_all(x, result['optimized_coeffs'])
    axes[1].plot(x, u_final, color='red', linewidth=2.5, linestyle='--', label='Final')

    axes[1].set_title('Waveform Convergence Trajectory')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u(x)')
    axes[1].grid(True)
    # 凡例が多すぎると見づらいので縮小表示するかお好みで調整
    axes[1].legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    path_dyn = output_dir / 'coefficient_dynamics.png'
    plt.savefig(path_dyn, dpi=300)
    plt.close()
    print(f"-> 係数ダイナミクスの診断グラフを保存しました: {path_dyn}")

def save_basis_adoption_heatmap(result: dict, output_dir: str | Path = "outputs"):
    """
    【可視化①：基底の採用率時系列ヒートマップ】
    世代ごとの各次数の基底の採用率（全個体の中でその基底がONになっている割合）を
    ヒートマップとして可視化・保存する
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lineage_records = result.get('lineage_records', [])
    if not lineage_records:
        print("-> 警告: lineage_records が見つからないため、基底採用率ヒートマップの作成をスキップします。")
        return

    # 世代ごとにマスクのリストをグループ化
    generations_dict = {}
    for record in lineage_records:
        gen = record['generation']
        if gen not in generations_dict:
            generations_dict[gen] = []
        generations_dict[gen].append(record['mask'])

    max_gen = max(generations_dict.keys())
    sample_mask = lineage_records[0]['mask']
    max_degree = len(sample_mask) - 1

    # 採用率行列の作成: 形状は (max_degree + 1, max_gen + 1)
    adoption_matrix = np.zeros((max_degree + 1, max_gen + 1))

    for gen in sorted(generations_dict.keys()):
        masks = np.array(generations_dict[gen]) # shape: (pop_size, max_degree + 1)
        # 各列（次数ごと）の平均を計算（1が立っている割合 = 採用率）
        adoption_rates = np.mean(masks, axis=0)
        adoption_matrix[:, gen] = adoption_rates

    # ヒートマップの描画
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(adoption_matrix, aspect='auto', origin='lower', cmap='Blues', vmin=0.0, vmax=1.0)
    
    fig.colorbar(cax, label='Adoption Rate (Probability)')
    ax.set_title("Basis Adoption Rate over Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Basis Degree (n)")
    
    # 整数目盛りに設定
    ax.set_xticks(range(max_gen + 1))
    ax.set_yticks(range(max_degree + 1))

    plt.tight_layout()
    path_out = output_dir / 'basis_adoption_heatmap.png'
    plt.savefig(path_out, dpi=300)
    plt.close()
    print(f"-> 成果物を保存しました: {path_out}")

def save_loss_distribution_plot(result: dict, output_dir: str | Path = "outputs"):
    """
    【可視化②：世代ごとのLoss分布の推移】
    各世代における全個体のLossの分布（中央値、25%-75%範囲、最小値）を
    対数スケールのバンドグラフとして可視化・保存する
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lineage_records = result.get('lineage_records', [])
    if not lineage_records:
        print("-> 警告: lineage_records が見つからないため、Loss分布グラフの作成をスキップします。")
        return

    # 世代ごとにLossを収集
    gen_losses = {}
    for record in lineage_records:
        gen = record['generation']
        loss = record['loss']
        # 無限大やNaNを除外して対数表示に備える
        if not np.isinf(loss) and not np.isnan(loss):
            if gen not in gen_losses:
                gen_losses[gen] = []
            gen_losses[gen].append(loss)

    generations = sorted(gen_losses.keys())
    medians = []
    p25s = []
    p75s = []
    mins = []
    maxs = []

    for gen in generations:
        losses = np.array(gen_losses[gen])
        medians.append(np.median(losses))
        p25s.append(np.percentile(losses, 25))
        p75s.append(np.percentile(losses, 75))
        mins.append(np.min(losses))
        maxs.append(np.max(losses))

    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gens = np.array(generations)
    medians = np.array(medians)
    p25s = np.array(p25s)
    p75s = np.array(p75s)
    mins = np.array(mins)
    maxs = np.array(maxs)

    # IQR（25%〜75%）のバンド
    ax.fill_between(gens, p25s, p75s, color='tab:blue', alpha=0.3, label='IQR (25% - 75%)')
    # 最小〜最大のバンド（または点線）
    ax.fill_between(gens, mins, maxs, color='tab:blue', alpha=0.1, label='Min - Max Range')
    
    # 中央値と最良値（最小値）の推移線
    ax.plot(gens, medians, color='tab:blue', linewidth=2, label='Median Loss')
    ax.plot(gens, mins, color='tab:red', linestyle='--', linewidth=1.5, label='Best Loss (Min)')

    ax.set_yscale('log')
    ax.set_title("Population Loss Distribution over Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Loss (log scale)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc='upper right')

    plt.tight_layout()
    path_out = output_dir / 'loss_distribution_band.png'
    plt.savefig(path_out, dpi=300)
    plt.close()
    print(f"-> 成果物を保存しました: {path_out}")
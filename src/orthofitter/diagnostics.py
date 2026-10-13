import numpy as np

def compute_spatial_entropy(residuals: np.ndarray) -> float:
    """
    【診断軸1】残差の空間エントロピー H_res
    残差の絶対値の空間的分布を確率分布とみなしてシャノン・エントロピーを算出する。
    """
    abs_res = np.abs(residuals)
    total_sum = np.sum(abs_res)
    
    if total_sum < 1e-12:
        return 0.0
    
    # 確率化
    p = abs_res / total_sum
    # 0除算を防ぐためのマスク
    p_safe = p[p > 0]
    
    entropy = -np.sum(p_safe * np.log(p_safe))
    return float(entropy)

def compute_jacobian_spectrum(catalog, x: np.ndarray, mask: np.ndarray) -> dict:
    """
    【診断軸2】ヤコビアン（デザインマトリックス）のスペクトル解析
    選択されている有効な基底群 G の特異値分解から条件数と有効ランクを計算する。
    """
    active_indices = np.where(mask == 1)[0]
    if len(active_indices) == 0:
        return {"condition_number": float('inf'), "effective_rank": 0, "max_singular_value": 0.0}

    # デザインマトリックス G の構築 (形状: [len(x), len(active_indices)])
    G = np.zeros((len(x), len(active_indices)))
    for idx, n in enumerate(active_indices):
        G[:, idx] = catalog.evaluate(n, x)

    # 特異値分解 (SVD)
    try:
        singular_values = np.linalg.svd(G, compute_uv=False)
    except np.linalg.LinAlgError:
        return {"condition_number": float('inf'), "effective_rank": 0, "max_singular_value": 0.0}

    s_max = singular_values[0]
    s_min = singular_values[-1]
    
    # 条件数の計算 (ゼロ割対策)
    if s_min < 1e-12:
        cond_num = float('inf')
    else:
        cond_num = float(s_max / s_min)

    # 有効ランク（特異値の相対的な大きさが閾値を超える数）
    threshold = s_max * 1e-6
    effective_rank = int(np.sum(singular_values > threshold))

    return {
        "condition_number": cond_num,
        "effective_rank": effective_rank,
        "max_singular_value": float(s_max),
        "min_singular_value": float(s_min)
    }

def compute_spectral_decay(coeffs: np.ndarray, critical_ratio: float = 0.6) -> float:
    """
    【診断軸3】スペクトル減衰率 γ
    係数ベクトル全体のエネルギーに対して、高周波領域（全体の上位40%の次数）が占める割合を計算する。
    """
    if len(coeffs) == 0:
        return 0.0

    energies = coeffs ** 2
    total_energy = np.sum(energies)
    
    if total_energy < 1e-12:
        return 0.0

    dim = len(coeffs)
    split_idx = int(dim * critical_ratio)
    
    # 高周波成分のエネルギー
    high_freq_energy = np.sum(energies[split_idx:])
    
    return float(high_freq_energy / total_energy)

def compute_step_efficiency(prev_coeffs: np.ndarray, curr_coeffs: np.ndarray, prev_loss: float, curr_loss: float) -> dict:
    """
    【診断軸4】探索効率 (eta_step および 軌跡のなす角 cos_phi の準備)
    パラメータの移動距離あたりの損失改善量を算出する。
    """
    if prev_coeffs is None or curr_coeffs is None:
        return {"eta_step": 0.0, "param_distance": 0.0}

    param_distance = np.linalg.norm(curr_coeffs - prev_coeffs)
    loss_diff = prev_loss - curr_loss  # 正であれば改善

    if param_distance < 1e-12:
        eta_step = 0.0
    else:
        eta_step = float(loss_diff / param_distance)

    return {
        "eta_step": eta_step,
        "param_distance": float(param_distance),
        "loss_diff": float(loss_diff)
    }
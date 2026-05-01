# Code developed by : Praveen + Prajin

"""
SON Clustering Recovery under Anisotropic (Ellipsoidal) Distributions
======================================================================
Prompt: Experimental Protocol – April 24, 2026
Theory: Dunlap & Mourrat, "Local Versions of Sum-of-Norms Clustering"
        arXiv:2109.09589v3

Objective
---------
  f(x) = (1/2) Σ_i ‖x_i − a_i‖²  +  λ Σ_{i<j} w_ij ‖x_i − x_j‖_2

Localized weight:  w_ij = γ^(d+1) exp(−γ ‖a_i − a_j‖)
                   γ = N^{3/(4d)}   (Theorem 1.2 scaling law)

ADMM updates (vectorised)
-------------------------
  V_ij = x_i − x_j + U_ij
  Z_ij = max(1 − λ w_ij / (ρ ‖V_ij‖), 0) · V_ij      [soft-threshold]
  x_i  = (a_i + ρ Σ_j (x_j + Z_ij − U_ij)) / (1 + ρ n)
  U_ij = U_ij + x_i − x_j − Z_ij

Metrics
-------
  s1  Core Recovery   : fraction of core points whose assigned
                        predicted cluster has majority from the
                        correct true cluster AND all core members
                        of true cluster k share the same predicted
                        cluster  → penalises BOTH fragmentation
                        (under-fusion) and contamination (over-fusion)
  s2  Total Accuracy  : global fraction correctly assigned (majority vote)
  s3  Component Distinctness: |{distinct true labels in pred_to_true}| / K

Four plots
----------
  Panel 1 – Data Landscape : a_i + 2-σ covariance ellipses
  Panel 2 – Solution Space : x*_i (fusion effect) at chosen λ
  Panel 3 – Heatmap        : s1 over (λ, σ) grid  → "Sweet Spot"
  Panel 4 – Cluster Path   : #clusters vs λ (log scale)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
from collections import Counter

# ═══════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SEED          = 7
N_MAIN        = 600        # points for panels 1 & 2
N_GRID        = 90         # smaller N for (λ,σ) grid search
N_PATH        = 100        # N for cluster-path sweep
D             = 2
K_TRUE        = 3

LAMBDA_MAIN   = 0.08       # λ for panels 1 & 2
SIGMA_BASE    = 1.0        # σ multiplier for main run
THETA         = 1.5        # core-radius: core_k = {‖a_i−μ_k‖ ≤ θ·σ_k}

RHO           = 1.0
MAX_ITER_MAIN = 250
MAX_ITER_FAST = 70
TOL_MAIN      = 1e-6
TOL_FAST      = 3e-3
CLUSTER_EPS   = 1e-3
USE_SPARSE    = True

# Grid-search ranges
LAMBDA_GRID  = np.logspace(-4, 1, 14)
SIGMA_MULTS  = np.array([0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.4, 3.2])

# Cluster-path
LAMBDA_PATH  = np.logspace(-3, 0.5, 22)

# ═══════════════════════════════════════════════════════════════
# 1.  ANISOTROPIC DATA GENERATION
#     Three elongated Gaussians in a tight triangle:
#       - high eccentricity  (10 : 1 eigenvalue ratio)
#       - elongated arms point toward each other  → convex hulls overlap
#       - centres only ~2 units apart
# ═══════════════════════════════════════════════════════════════
def rotation(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]])

def make_cov(angle_deg, lam_major, lam_minor, sigma_mult=1.0):
    R = rotation(np.radians(angle_deg))
    L = np.diag([lam_major * sigma_mult**2, lam_minor * sigma_mult**2])
    return R @ L @ R.T

# Tight triangle – elongated axes point inward
CLUSTER_DEFS = [
    #  centre                    angle   major  minor
    (np.array([ 0.00,  1.80]),    30,   0.70,  0.07),  # top
    (np.array([-1.56, -0.90]),   -30,   0.70,  0.07),  # bottom-left
    (np.array([ 1.56, -0.90]),    90,   0.70,  0.07),  # bottom-right
]

def generate_data(N, sigma_mult=1.0, seed=SEED):
    rng    = np.random.default_rng(seed)
    n_per  = N // K_TRUE
    counts = [n_per, n_per, N - 2 * n_per]
    A_list, L_list, means_list, covs_list = [], [], [], []
    for k, (mu, ang, lmaj, lmin) in enumerate(CLUSTER_DEFS):
        cov = make_cov(ang, lmaj, lmin, sigma_mult)
        pts = rng.multivariate_normal(mu, cov, counts[k])
        A_list.append(pts)
        L_list.append(np.full(counts[k], k, dtype=int))
        means_list.append(mu.copy())
        covs_list.append(cov)
    return (np.vstack(A_list), np.concatenate(L_list),
            np.array(means_list), covs_list)

# ═══════════════════════════════════════════════════════════════
# 2.  WEIGHT MATRIX
# ═══════════════════════════════════════════════════════════════
def build_weights(A, use_sparse=USE_SPARSE):
    N     = len(A)
    gamma = N ** (3.0 / (4.0 * D))
    dist  = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)
    W     = (gamma ** (D + 1)) * np.exp(-gamma * dist)
    if use_sparse:
        omega = (D + 4.0 / 3.0) * np.log(gamma) / gamma
        W[dist > omega] = 0.0
    np.fill_diagonal(W, 0.0)
    return W, gamma

# ═══════════════════════════════════════════════════════════════
# 3.  ADMM SOLVER  (vectorised)
# ═══════════════════════════════════════════════════════════════
def run_admm(A, W, lam, rho=RHO, max_iter=MAX_ITER_MAIN, tol=TOL_MAIN,
             X_init=None, verbose=False):
    N, D2 = A.shape
    X  = A.copy() if X_init is None else X_init.copy()
    Z  = np.zeros((N, N, D2))
    U  = np.zeros((N, N, D2))
    lwr = (lam * W) / rho
    for it in range(1, max_iter + 1):
        X_old = X.copy()
        V      = X[:, None, :] - X[None, :, :] + U
        V_norm = np.linalg.norm(V, axis=-1, keepdims=True)
        V_norm = np.where(V_norm == 0.0, 1.0, V_norm)
        Z      = np.maximum(1.0 - lwr[:, :, None] / V_norm, 0.0) * V
        Z      = 0.5 * (Z - Z.transpose(1, 0, 2))
        sum_x   = X.sum(axis=0, keepdims=True)
        sum_ZmU = (Z - U).sum(axis=1)
        X = (A + rho * (sum_x + sum_ZmU)) / (1.0 + rho * N)
        U = U + X[:, None, :] - X[None, :, :] - Z
        p_res = np.linalg.norm(X - X_old)
        d_res = rho * np.linalg.norm(
            (X[:, None, :] - X[None, :, :]) -
            (X_old[:, None, :] - X_old[None, :, :])
        )
        if verbose and (it % 50 == 0 or it == 1):
            print(f"  iter {it:4d}  primal={p_res:.2e}  dual={d_res:.2e}")
        if p_res < tol and d_res < tol and it > 10:
            if verbose:
                print(f"  Converged at iter {it}.")
            break
    return X

# ═══════════════════════════════════════════════════════════════
# 4.  CLUSTER EXTRACTION
# ═══════════════════════════════════════════════════════════════
def extract_clusters(X_star, eps=CLUSTER_EPS):
    N      = len(X_star)
    parent = np.arange(N)
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
    dx = cdist(X_star, X_star)
    r, c = np.where((dx < eps) & (dx > 0))
    for ri, ci in zip(r, c):
        union(ri, ci)
    cids  = np.array([find(i) for i in range(N)])
    roots = np.unique(cids)
    return np.searchsorted(roots, cids), len(roots)

# ═══════════════════════════════════════════════════════════════
# 5.  METRICS
#     s1 — Core Recovery
#          For each true cluster k:
#            • find the predicted cluster p* that the majority of
#              core-k points belong to
#            • fraction of core-k points that are in p*
#          s1 = mean over k
#          → equals 1 only when ALL cores are perfectly fused
#          → drops if cores are fragmented (small λ) or merged
#            with another cluster's core (large λ)
#     s2 — Total accuracy (majority-vote)
#     s3 — Component distinctness
# ═══════════════════════════════════════════════════════════════
def compute_metrics(pred, true_labels, A, means, covs, theta=THETA):
    N   = len(true_labels)
    # per-cluster RMS radius σ_k = sqrt(tr(Σ_k) / d)
    sigma_k = [np.sqrt(np.trace(c) / D) for c in covs]

    # --- s1 : core recovery ---
    s1_per_cluster = []
    for k in range(K_TRUE):
        dist_k    = np.linalg.norm(A - means[k], axis=1)
        core_mask = (true_labels == k) & (dist_k <= theta * sigma_k[k])
        if core_mask.sum() == 0:
            continue
        core_preds     = pred[core_mask]
        majority_pred  = Counter(core_preds).most_common(1)[0][0]
        purity         = (core_preds == majority_pred).mean()
        s1_per_cluster.append(purity)
    s1 = float(np.mean(s1_per_cluster)) if s1_per_cluster else 0.0

    # --- s2 : total accuracy ---
    p2t = {}
    for pid in np.unique(pred):
        mask    = pred == pid
        p2t[pid] = Counter(true_labels[mask]).most_common(1)[0][0]
    correct = sum(p2t[pred[i]] == true_labels[i] for i in range(N))
    s2      = correct / N

    # --- s3 : component distinctness ---
    s3 = len(set(p2t.values())) / K_TRUE

    return s1, s2, s3

# ═══════════════════════════════════════════════════════════════
# 6.  ELLIPSE HELPER
# ═══════════════════════════════════════════════════════════════
def draw_ellipse(ax, mu, cov, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals  = vals[order];  vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=mu, width=w, height=h, angle=angle, **kwargs))

# ═══════════════════════════════════════════════════════════════
# 7.  BUILD MAIN DATA
# ═══════════════════════════════════════════════════════════════
A_main, labels_main, means_main, covs_main = generate_data(N_MAIN, SIGMA_BASE)
W_main, gamma_main = build_weights(A_main)

print("=" * 64)
print(f"SON Clustering – Anisotropic Recovery Experiment")
print(f"N={N_MAIN}  D={D}  K={K_TRUE}  γ={gamma_main:.3f}  λ_main={LAMBDA_MAIN}")
print(f"Non-zero weights (main): {int(np.sum(W_main > 0))} / {N_MAIN**2}")
print("=" * 64)

# ═══════════════════════════════════════════════════════════════
# 8.  (λ, σ) GRID SEARCH
# ═══════════════════════════════════════════════════════════════
print(f"\nGrid search: {len(LAMBDA_GRID)} λ × {len(SIGMA_MULTS)} σ …")
S1 = np.zeros((len(SIGMA_MULTS), len(LAMBDA_GRID)))
S2 = np.zeros_like(S1)
S3 = np.zeros_like(S1)

for si, sm in enumerate(SIGMA_MULTS):
    A_g, L_g, mu_g, cov_g = generate_data(N_GRID, sm, seed=SEED + si * 13)
    W_g, _ = build_weights(A_g)
    X_warm = None
    for li, lam in enumerate(LAMBDA_GRID):
        Xs = run_admm(A_g, W_g, lam, rho=RHO,
                      max_iter=MAX_ITER_FAST, tol=TOL_FAST,
                      X_init=X_warm, verbose=False)
        pred_g, _ = extract_clusters(Xs)
        s1v, s2v, s3v = compute_metrics(pred_g, L_g, A_g, mu_g, cov_g)
        S1[si, li] = s1v
        S2[si, li] = s2v
        S3[si, li] = s3v
        X_warm = Xs.copy()
    print(f"  σ×{sm:.2f}  done  |  best s1={S1[si].max():.3f}"
          f"  @λ={LAMBDA_GRID[S1[si].argmax()]:.5f}")

print("Grid search complete.\n")
best_idx = np.unravel_index(S1.argmax(), S1.shape)
print(f"Sweet spot: λ={LAMBDA_GRID[best_idx[1]]:.5f}  "
      f"σ×{SIGMA_MULTS[best_idx[0]]:.2f}  s1={S1[best_idx]:.3f}")

# ═══════════════════════════════════════════════════════════════
# 9.  CLUSTER-PATH SWEEP
# ═══════════════════════════════════════════════════════════════
print(f"\nCluster-path sweep (N={N_PATH}) …")
A_path, L_path, mu_path, cov_path = generate_data(N_PATH, 1.0, seed=SEED)
W_path, _ = build_weights(A_path)
path_n, X_warm = [], None
for lam in LAMBDA_PATH:
    Xs = run_admm(A_path, W_path, lam, rho=RHO,
                  max_iter=MAX_ITER_FAST, tol=TOL_FAST,
                  X_init=X_warm, verbose=False)
    _, n_c = extract_clusters(Xs)
    path_n.append(n_c)
    X_warm = Xs.copy()
    print(f"  λ={lam:.5f}  →  {n_c:3d} clusters")
print("Path sweep done.\n")

# ═══════════════════════════════════════════════════════════════
# 10. MAIN ADMM RUN
# ═══════════════════════════════════════════════════════════════
print(f"Main ADMM run: N={N_MAIN}  λ={LAMBDA_MAIN} …")
X_star = run_admm(A_main, W_main, LAMBDA_MAIN, rho=RHO,
                  max_iter=MAX_ITER_MAIN, tol=TOL_MAIN, verbose=True)
pred_main, n_found = extract_clusters(X_star)
s1_m, s2_m, s3_m  = compute_metrics(pred_main, labels_main,
                                      A_main, means_main, covs_main)
print()
print("=" * 64)
print(f"  s1 (core recovery)        = {s1_m:.4f}")
print(f"  s2 (total accuracy)       = {s2_m:.4f}")
print(f"  s3 (component distinct.)  = {s3_m:.4f}  [{n_found} groups found]")
print("=" * 64)

# ═══════════════════════════════════════════════════════════════
# 11. FOUR-PANEL FIGURE
# ═══════════════════════════════════════════════════════════════
PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261', '#264653', '#E9C46A']

cmap_heat = LinearSegmentedColormap.from_list(
    'recovery',
    ['#03045e', '#0077b6', '#00b4d8', '#90e0ef',
     '#caf0f8', '#ffffff', '#ffe8d6', '#f4a261', '#e76f51'],
    N=256
)

fig = plt.figure(figsize=(17, 13))
fig.patch.set_facecolor('#f0f4f8')
gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30,
                        left=0.07, right=0.96,
                        top=0.91, bottom=0.07)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

fig.suptitle(
    "SON Clustering — Anisotropic (Ellipsoidal) Recovery Experiment\n"
    r"Localized weight $w_{ij}=\gamma^{d+1}e^{-\gamma\|a_i-a_j\|}$,  "
    rf"$\gamma = N^{{3/(4d)}} = {gamma_main:.2f}$,  "
    rf"$\lambda_\mathrm{{main}} = {LAMBDA_MAIN}$",
    fontsize=14, fontweight='bold', y=0.97,
    color='#1a1a2e'
)

# ─── PANEL 1 : Data Landscape ───────────────────────────────
for k in range(K_TRUE):
    mk = labels_main == k
    ax1.scatter(A_main[mk, 0], A_main[mk, 1],
                c=PALETTE[k], s=11, alpha=0.62, linewidths=0,
                label=f"Cluster {k+1}  (n={mk.sum()})", zorder=3)
    # filled 2-σ ellipse (semi-transparent)
    draw_ellipse(ax1, means_main[k], covs_main[k], n_std=2.0,
                 edgecolor=PALETTE[k], facecolor=PALETTE[k],
                 alpha=0.10, linewidth=0, zorder=1)
    # outline
    draw_ellipse(ax1, means_main[k], covs_main[k], n_std=2.0,
                 edgecolor=PALETTE[k], facecolor='none',
                 linewidth=2.0, linestyle='--', zorder=2)
    # centre cross
    ax1.plot(*means_main[k], '+', color=PALETTE[k],
             markersize=12, markeredgewidth=2.5, zorder=5)

ax1.set_facecolor('#fdfdff')
ax1.set_title(
    f"Panel 1 — Data Landscape  (N={N_MAIN}, σ×{SIGMA_BASE:.1f})\n"
    "Dashed ellipses = 2σ covariance  |  + = centroid",
    fontsize=9.5, fontweight='bold', pad=6
)
ax1.set_xlabel("$x_1$", fontsize=9); ax1.set_ylabel("$x_2$", fontsize=9)
ax1.legend(fontsize=8, markerscale=1.8, loc='upper right', framealpha=0.85)
ax1.set_aspect('equal', adjustable='datalim')
ax1.grid(True, linestyle='--', alpha=0.22)

# ─── PANEL 2 : Solution Space ────────────────────────────────
for pid in np.unique(pred_main):
    mk  = pred_main == pid
    col = PALETTE[pid % len(PALETTE)]
    ax2.scatter(X_star[mk, 0], X_star[mk, 1],
                c=col, s=11, alpha=0.78, linewidths=0, zorder=3)

# reference ellipses (dashed outline)
for k in range(K_TRUE):
    draw_ellipse(ax2, means_main[k], covs_main[k], n_std=2.0,
                 edgecolor=PALETTE[k], facecolor='none',
                 linewidth=1.4, linestyle=':', alpha=0.55, zorder=2)

# unique representative stars
reps = np.array([X_star[pred_main == pid].mean(axis=0)
                 for pid in np.unique(pred_main)])
ax2.scatter(reps[:, 0], reps[:, 1], c='black',
            marker='*', s=150, zorder=6,
            label=f"Representatives  ({n_found} groups)")

ax2.set_facecolor('#fdfdff')
ax2.set_title(
    f"Panel 2 — Solution Space  $x^*_i$  (λ={LAMBDA_MAIN})\n"
    f"s1={s1_m:.3f}   s2={s2_m:.3f}   s3={s3_m:.3f}   "
    f"[{n_found} groups found]",
    fontsize=9.5, fontweight='bold', pad=6
)
ax2.set_xlabel("$x_1$", fontsize=9); ax2.set_ylabel("$x_2$", fontsize=9)
ax2.legend(fontsize=8, loc='upper right', framealpha=0.85)
ax2.set_aspect('equal', adjustable='datalim')
ax2.grid(True, linestyle='--', alpha=0.22)

# ─── PANEL 3 : Heatmap ──────────────────────────────────────
im = ax3.imshow(
    S1, aspect='auto', origin='lower',
    cmap=cmap_heat, vmin=0.0, vmax=1.0,
    extent=[0, len(LAMBDA_GRID), 0, len(SIGMA_MULTS)]
)

# cell annotations
for si in range(len(SIGMA_MULTS)):
    for li in range(len(LAMBDA_GRID)):
        v = S1[si, li]
        tc = 'white' if v < 0.45 or v > 0.75 else '#1a1a2e'
        ax3.text(li + 0.5, si + 0.5, f"{v:.2f}",
                 ha='center', va='center', fontsize=6.0,
                 color=tc, fontweight='bold')

# star on best cell
ax3.plot(best_idx[1] + 0.5, best_idx[0] + 0.5,
         '*', color='gold', markersize=14,
         markeredgecolor='black', markeredgewidth=0.8, zorder=5)

# draw a rectangle outline around sweet-spot region
# (cells with s1 ≥ 0.85 * max)
thresh = 0.85 * S1.max()
for si in range(len(SIGMA_MULTS)):
    for li in range(len(LAMBDA_GRID)):
        if S1[si, li] >= thresh:
            ax3.add_patch(plt.Rectangle(
                (li, si), 1, 1,
                fill=False, edgecolor='gold',
                linewidth=1.4, zorder=4
            ))

lam_labels = [f"{v:.0e}" for v in LAMBDA_GRID]
sig_labels  = [f"{v:.1f}" for v in SIGMA_MULTS]
ax3.set_xticks(np.arange(len(LAMBDA_GRID)) + 0.5)
ax3.set_xticklabels(lam_labels, rotation=55, ha='right', fontsize=6.2)
ax3.set_yticks(np.arange(len(SIGMA_MULTS)) + 0.5)
ax3.set_yticklabels(sig_labels, fontsize=7.5)
ax3.set_xlabel("Regularisation  λ  →", fontsize=9)
ax3.set_ylabel("σ multiplier  →", fontsize=9)
ax3.set_title(
    "Panel 3 — Heatmap: s1 Core Recovery  over  (λ, σ) Grid\n"
    "★ = global sweet spot   |   gold border = high-recovery zone",
    fontsize=9.5, fontweight='bold', pad=6
)
cb = plt.colorbar(im, ax=ax3, fraction=0.034, pad=0.02)
cb.set_label("s1 (core recovery)", fontsize=8)
cb.ax.tick_params(labelsize=7.5)

# ─── PANEL 4 : Cluster Path ──────────────────────────────────
ax4.step(LAMBDA_PATH, path_n, where='post',
         color='#264653', linewidth=2.3, zorder=4, label='# clusters')
ax4.scatter(LAMBDA_PATH, path_n,
            color='#E9C46A', edgecolors='#264653',
            s=52, zorder=5, linewidths=0.9)
ax4.fill_between(LAMBDA_PATH, path_n, step='post',
                 alpha=0.17, color='#457B9D')

ax4.axvline(LAMBDA_MAIN, color='black', linestyle=':', linewidth=1.8,
            label=f'λ_main = {LAMBDA_MAIN}')
ax4.axhline(K_TRUE, color='#E63946', linestyle='--', linewidth=1.7,
            label=f'True K = {K_TRUE}', zorder=3)

# annotate operating point
idx_op = np.argmin(np.abs(LAMBDA_PATH - LAMBDA_MAIN))
k_op   = path_n[idx_op]
y_rng  = max(max(path_n) - K_TRUE + 2, 5)

ax4.annotate(
    f"λ={LAMBDA_MAIN}\nK={k_op}",
    xy=(LAMBDA_MAIN, k_op),
    xytext=(LAMBDA_MAIN * 6, k_op + y_rng * 0.3),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
    fontsize=8.5, color='#1a1a2e',
    bbox=dict(boxstyle='round,pad=0.35', fc='white',
              ec='#457B9D', alpha=0.90)
)

# annotate first time K_TRUE is hit
for i, kv in enumerate(path_n):
    if kv == K_TRUE:
        ax4.annotate(
            f"K={K_TRUE} first hit\nλ={LAMBDA_PATH[i]:.3f}",
            xy=(LAMBDA_PATH[i], K_TRUE),
            xytext=(LAMBDA_PATH[i] * 0.25, K_TRUE + y_rng * 0.4),
            arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.0),
            fontsize=8, color='#E63946',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='#E63946', alpha=0.90)
        )
        break

ax4.set_xscale('log')
ax4.set_xlabel("Regularisation parameter  λ  (log scale)", fontsize=9)
ax4.set_ylabel("Number of distinct clusters", fontsize=9)
ax4.set_title(
    "Panel 4 — Cluster Path: #clusters vs λ\n"
    "(phase transitions / agglomeration)",
    fontsize=9.5, fontweight='bold', pad=6
)
ax4.legend(fontsize=8.5, loc='upper right', framealpha=0.90)
ax4.grid(True, which='both', linestyle='--', alpha=0.22)
ax4.set_facecolor('#fdfdff')

# ─── Save ────────────────────────────────────────────────────
out_fig = "/mnt/user-data/outputs/son_anisotropic_recovery.png"
fig.savefig(out_fig, dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"\nFigure saved → {out_fig}")
plt.show()
print("Done.")

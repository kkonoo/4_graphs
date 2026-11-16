"""05 §2 — Louvain vs Leiden, 정답을 아는 그래프에서 비교.

확률 블록 모형(SBM)으로 커뮤니티 6개(각 80노드)를 만들고, 커뮤니티 사이
연결 확률만 올려가며 두 알고리즘을 시드 20개로 돌린다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
import igraph as ig
import leidenalg as la
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

n_comm, size, p_in = 6, 80, 0.10
p_outs = [0.004, 0.010, 0.018, 0.026]
SEEDS = 20


def labels(G, parts):
    d = {}
    for i, c in enumerate(parts):
        for v in c:
            d[v] = i
    return np.array([d[v] for v in G.nodes()])


res = {k: [] for k in ("lvQ", "ldQ", "lvARI", "ldARI", "lvNMI", "ldNMI",
                       "lvK", "ldK")}
for p_out in p_outs:
    P = np.full((n_comm, n_comm), p_out); np.fill_diagonal(P, p_in)
    G = nx.stochastic_block_model([size] * n_comm, P, seed=1)
    truth = np.concatenate([[i] * size for i in range(n_comm)])
    g = ig.Graph.from_networkx(G)

    lvQ, ldQ, lvL, ldL, lvK, ldK = [], [], [], [], [], []
    for s in range(SEEDS):
        pl = louvain_communities(G, seed=s)
        lvQ.append(modularity(G, pl)); lvL.append(labels(G, pl)); lvK.append(len(pl))
        pe = la.find_partition(g, la.ModularityVertexPartition, seed=s,
                               n_iterations=2)
        ldQ.append(g.modularity(pe.membership))
        ldL.append(np.array(pe.membership)); ldK.append(len(pe))

    pair = lambda L: np.mean([adjusted_rand_score(L[i], L[j])
                              for i in range(len(L)) for j in range(i + 1, len(L))])
    res["lvQ"].append(np.mean(lvQ));  res["ldQ"].append(np.mean(ldQ))
    res["lvARI"].append(pair(lvL));   res["ldARI"].append(pair(ldL))
    res["lvNMI"].append(np.mean([normalized_mutual_info_score(truth, l) for l in lvL]))
    res["ldNMI"].append(np.mean([normalized_mutual_info_score(truth, l) for l in ldL]))
    res["lvK"].append(np.mean(lvK));  res["ldK"].append(np.mean(ldK))
    print(f"p_out={p_out}: Q {np.mean(lvQ):.4f}/{np.mean(ldQ):.4f} "
          f"ARI {pair(lvL):.3f}/{pair(ldL):.3f} "
          f"NMI {res['lvNMI'][-1]:.3f}/{res['ldNMI'][-1]:.3f} "
          f"k {np.mean(lvK):.1f}/{np.mean(ldK):.1f}")

# ── 그리기 ────────────────────────────────────────────────────────────────
LV, LD = CAT[1], CAT[0]                 # 나란한 두 방법: 강조색 없음
plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})

x = np.arange(len(p_outs)); w = 0.36
panels = [("① 정답과의 일치도 (NMI)", "lvNMI", "ldNMI"),
          ("② 시드 간 재현성 (ARI)", "lvARI", "ldARI"),
          ("③ 찾은 커뮤니티 수", "lvK", "ldK")]

fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.4))
fig.subplots_adjust(left=0.065, right=0.995, wspace=0.28)
for ax, (name, kl, kd) in zip(axes, panels):
    ax.bar(x - w/2, res[kl], w, color=LV, label="Louvain", zorder=3)
    ax.bar(x + w/2, res[kd], w, color=LD, label="Leiden", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.3f}" for p in p_outs])
    ax.tick_params(axis="x", length=0, pad=5)
    ax.text(0.5, 1.05, name, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT)
    if kl == "lvK":
        ax.axhline(n_comm, color=TEXT2, ls=(0, (4, 3)), lw=1.3, zorder=4)
        ax.text(0.5, n_comm + 0.25, "정답 6개", ha="center", va="bottom",
                fontsize=11, color=TEXT2, zorder=5)
        ax.set_ylim(0, 8.3); ax.set_yticks([0, 2, 4, 6, 8])
    else:
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.55, len(p_outs) - 0.45)

axes[1].set_xlabel("커뮤니티 사이 연결 확률 (어려워짐 →)", fontsize=11.5, labelpad=7)
handles = [mp.Patch(color=LV, label="Louvain"), mp.Patch(color=LD, label="Leiden")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 1.0), ncol=2,
           frameon=False, fontsize=11.5, handlelength=1.2, columnspacing=2.0)

save(fig, img("c05", "louvain_leiden.png"))

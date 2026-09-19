"""05 §2 — Louvain vs Leiden, 정답을 아는 그래프에서 비교.

확률 블록 모형(SBM)으로 커뮤니티 6개(각 80노드)를 만들고, 커뮤니티 사이
연결 확률만 올려가며 두 알고리즘을 시드 20개로 돌린다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
import igraph as ig
import leidenalg as la
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

OUT = outdir("c05")

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

x = np.arange(len(p_outs)); w = 0.36
panels = [("정답과의 일치도 (NMI)", "lvNMI", "ldNMI"),
          ("시드 간 재현성 (ARI)", "lvARI", "ldARI"),
          ("찾은 커뮤니티 수", "lvK", "ldK")]

fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), gridspec_kw={"wspace": 0.26})
for ax, (title, kl, kd) in zip(axes, panels):
    ax.bar(x - w/2, res[kl], w, color=GREY, label="Louvain")
    ax.bar(x + w/2, res[kd], w, color=TEAL, label="Leiden")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.3f}" for p in p_outs])
    ax.set_xlabel("커뮤니티 사이 연결 확률 (어려워짐 →)")
    ax.set_title(title, color=NAVY, fontsize=12)
    if kl == "lvK":
        ax.axhline(n_comm, color=RED, ls="--", lw=1.4, label="정답 6개")
    ax.legend(frameon=False, fontsize=9.5)

fig.suptitle("커뮤니티 6개(각 80노드) · 시드 20개 평균 — 어려워질수록 격차가 벌어진다",
             color=NAVY, fontsize=13, fontweight="bold", y=1.03)
fig.savefig(os.path.join(OUT, "louvain_leiden.png"))
print("wrote", os.path.join(OUT, "louvain_leiden.png"))

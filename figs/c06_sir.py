"""06 §4 — SIR 확산과 개입 전략.

허브가 있는 네트워크에서는 10%를 무작위로 빼는 것과
차수 상위 10%를 빼는 것의 결과가 전혀 다르다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx

OUT = outdir("c06")

G = nx.powerlaw_cluster_graph(600, 3, 0.3, seed=2)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
N = len(G)
deg = dict(G.degree())
btw = nx.betweenness_centrality(G)
k = int(0.1 * N)

strategies = [
    ("제거 없음", [], NAVY),
    ("무작위 10%", list(np.random.default_rng(1).choice(list(G.nodes()), k, replace=False)), BLUE),
    ("차수 상위 10%", sorted(deg, key=deg.get, reverse=True)[:k], ORANGE),
    ("매개 상위 10%", sorted(btw, key=btw.get, reverse=True)[:k], TEAL),
]


def sir_curve(H, beta=0.06, gamma=0.1, n_seed=3, reps=60, tmax=120, seed=7):
    """각 시점의 누적 감염 비율(전체 N 대비)의 평균 곡선."""
    rng = np.random.default_rng(seed)
    nodes = list(H.nodes())
    curves = np.zeros((reps, tmax))
    for r in range(reps):
        if len(nodes) < n_seed:
            continue
        inf = set(rng.choice(nodes, n_seed, replace=False))
        rec = set()
        for t in range(tmax):
            new = set()
            for u in inf:
                for v in H.neighbors(u):
                    if v not in inf and v not in rec and rng.random() < beta:
                        new.add(v)
            rec |= {u for u in inf if rng.random() < gamma}
            inf = (inf | new) - rec
            curves[r, t] = (len(inf) + len(rec)) / N
            if not inf:
                curves[r, t:] = curves[r, t]
                break
    return curves.mean(0)


fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7),
                         gridspec_kw={"wspace": 0.26, "width_ratios": [1.25, 1]})

finals = []
for name, removed, col in strategies:
    H = G.copy(); H.remove_nodes_from(removed)
    c = sir_curve(H)
    finals.append((name, c[-1], col))
    axes[0].plot(c, color=col, lw=2.2, label=f"{name}  (최종 {c[-1]:.2f})")
    print(f"  {name:<14} 최종 유행 규모 {c[-1]:.3f}")

axes[0].set_xlabel("시간"); axes[0].set_ylabel("누적 감염 비율 (전체 대비)")
axes[0].set_ylim(0, 1)
axes[0].set_title("같은 10%를 빼도 누구를 빼느냐가 전부", color=NAVY, fontsize=12)
axes[0].legend(frameon=False, fontsize=10)

ax = axes[1]
ax.bar([f[0].replace(" ", "\n") for f in finals], [f[1] for f in finals],
       color=[f[2] for f in finals])
for i, f in enumerate(finals):
    ax.text(i, f[1] + 0.02, f"{f[1]:.2f}", ha="center", fontsize=11,
            fontweight="bold", color=f[2])
ax.set_ylim(0, 1)
ax.set_ylabel("최종 유행 규모")
ax.set_title("노드 10% 제거 — 전략별 결과", color=NAVY, fontsize=12)

fig.suptitle(f"SIR 시뮬레이션 (노드 {N}, β=0.06, γ=0.1, 60회 평균)",
             color=NAVY, fontsize=13, fontweight="bold", y=1.03)
fig.savefig(os.path.join(OUT, "sir.png"))
print("wrote", os.path.join(OUT, "sir.png"))

"""06 §4 — SIR 확산과 개입 전략.

허브가 있는 네트워크에서는 10%를 무작위로 빼는 것과
차수 상위 10%를 빼는 것의 결과가 전혀 다르다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx

NEUTRAL = "#9AA1AB"

G = nx.powerlaw_cluster_graph(600, 3, 0.3, seed=2)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
N = len(G)
deg = dict(G.degree())
btw = nx.betweenness_centrality(G)
k = int(0.1 * N)

# 제거 없음 = 기준선(무채색), 나머지 세 전략은 나란한 항목 → CAT
strategies = [
    ("제거 없음", [], NEUTRAL),
    ("무작위 10%", list(np.random.default_rng(1).choice(list(G.nodes()), k, replace=False)), CAT[1]),
    ("차수 상위 10%", sorted(deg, key=deg.get, reverse=True)[:k], CAT[0]),
    ("매개 상위 10%", sorted(btw, key=btw.get, reverse=True)[:k], CAT[2]),
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


plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8),
                         gridspec_kw={"wspace": 0.32, "width_ratios": [1.2, 1]})

finals, curves = [], []
for name, removed, col in strategies:
    H = G.copy(); H.remove_nodes_from(removed)
    c = sir_curve(H)
    finals.append((name, c[-1], col))
    curves.append(c)
    print(f"  {name:<14} 최종 유행 규모 {c[-1]:.3f}")
print(f"  노드 수 N = {N}, 제거 k = {k}")

# ── ① 시간에 따른 누적 감염 ─────────────────────────────────────────
ax = axes[0]
for (name, _, col), c in zip(strategies, curves):
    ax.plot(c, color=col, lw=2.2, zorder=3)
lab_y = {"제거 없음": 0.865, "무작위 10%": 0.635, "매개 상위 10%": 0.185, "차수 상위 10%": 0.11}
for name, _, col in strategies:
    ax.text(118, lab_y[name], name, color=shade(col, 0.15) if col == NEUTRAL else col,
            fontsize=11.5, fontweight="bold", ha="right", va="center")
ax.set_xlabel("시간", fontsize=11.5)
ax.set_ylabel("누적 감염 비율 (전체 대비)", fontsize=11.5)
ax.set_ylim(0, 1); ax.set_xlim(-3, 122)
ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax.grid(axis="y", color=LINE, lw=0.8, zorder=0)
ax.text(0.0, 1.05, "① 시간에 따른 누적 감염", transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12.5, fontweight="bold", color=TEXT)

# ── ② 전략별 최종 유행 규모 ─────────────────────────────────────────
ax = axes[1]
ticklabs = ["제거\n없음", "무작위\n10%", "차수\n상위\n10%", "매개\n상위\n10%"]
ax.bar(range(len(finals)), [f[1] for f in finals], color=[f[2] for f in finals], width=0.68, zorder=3)
for i, f in enumerate(finals):
    ax.text(i, f[1] + 0.025, f"{f[1]:.2f}", ha="center", va="bottom", fontsize=11.5,
            fontweight="bold", color=TEXT)
ax.set_xticks(range(len(finals)))
ax.set_xticklabels(ticklabs, fontsize=11, linespacing=1.2, va="top")
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 1)
ax.set_ylabel("최종 유행 규모", fontsize=11.5)
ax.grid(axis="y", color=LINE, lw=0.8, zorder=0)
ax.text(0.0, 1.05, "② 전략별 최종 유행 규모", transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12.5, fontweight="bold", color=TEXT)

save(fig, img("c06", "sir.png"))

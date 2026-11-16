"""11 §2-2 — node2vec의 q가 걸음의 반경을 바꾼다.

평면 kNN 그래프(점 600개, k = 6)에서 같은 출발점의 20걸음 × 3번 (q = 0.25 vs 4),
그리고 모든 노드에서 5번씩 걸었을 때 t걸음 뒤 출발점과의 평균 최단거리 (p = 1).
계산은 figs/c11_embedding.py의 '그림 1' 블록과 같고, 곡선 숫자는 cache/c11_walk_reach.csv에서 읽는다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c11_embedding import walks, spatial_knn, CACHE
from medfig import *
import pandas as pd
from matplotlib.collections import LineCollection

f1 = os.path.join(CACHE, "c11_walk_reach.csv")
r = pd.read_csv(f1)
G, P = spatial_knn()
start = int(np.argmin(((P - 0.5) ** 2).sum(1)))

QS = [0.25, 0.5, 1, 2, 4]
QC = dict(zip(QS, ramp(5)))            # q는 순서가 있는 값 → 테마 그라데이션
NEUTRAL = "#9AA1AB"

fig = plt.figure(figsize=(8.4, 5.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], hspace=0.32, wspace=0.2)
ax_nets = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
ax_line = fig.add_subplot(gs[:, 1])

segs = [P[[u, v]] for u, v in G.edges()]
for ax, q, mark in [(ax_nets[0], 0.25, "①"), (ax_nets[1], 4, "②")]:
    c = QC[q]
    ax.add_collection(LineCollection(segs, colors=LINE, linewidths=0.6, zorder=1))
    ax.scatter(*P.T, s=4, color=NEUTRAL, linewidths=0, zorder=2)
    rng = np.random.default_rng(1)
    for k in range(3):                                # 같은 출발점에서 20걸음 x 3번
        w = walks(G.subgraph(G.nodes), 1, q, n_walks=1, length=21, seed=10 + k)
        w = [x for x in w if x[0] == start][0]
        ax.plot(*P[w].T, color=c, lw=1.8, alpha=0.9, zorder=3, solid_capstyle="round")
    ax.scatter(*P[start], s=70, color=TEXT, zorder=4, edgecolor="white", linewidths=1.2)
    h20 = r[(r.q == q) & (r.t == 20)].hop.iloc[0]
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, 1.03, f"{mark} q = {q}", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12.5, fontweight="bold", color=TEXT)
    ax.text(0.5, -0.03, f"20걸음 뒤 평균 {h20:.1f} hop", transform=ax.transAxes, ha="center",
            va="top", fontsize=11, color=c, fontweight="bold")

ax = ax_line
for q in QS:
    s_ = r[r.q == q]
    ax.plot(s_.t, s_.hop, color=QC[q], lw=2, marker="o", ms=3.5, zorder=3)
    y20 = s_[s_.t == 20].hop.iloc[0]
    ax.text(20.6, y20, f"q = {q}", color=QC[q], fontsize=11, fontweight="bold", va="center")
ax.set_xlim(0.3, 23.6)
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_ylim(0.8, 5.2)
ax.set_yticks([1, 2, 3, 4, 5])
ax.tick_params(labelsize=11)
ax.grid(axis="y", color=LINE, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("걸음 수 t", fontsize=11.5)
ax.set_ylabel("출발점과의 최단거리 (hop, 평균)", fontsize=11.5)
ax.text(1.2, 5.0, "p = 1 고정", fontsize=11, color=TEXT2, va="center")
ax.text(0.5, 1.012, "③ t걸음 뒤 출발점과의 거리", transform=ax.transAxes, ha="center", va="bottom",
        fontsize=12.5, fontweight="bold", color=TEXT)

save(fig, img("c11", "walks.png"))
print(r[r.t.isin([5, 10, 20])].pivot(index="t", columns="q", values="hop").round(2))

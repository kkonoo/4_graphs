"""04 §1 — small-world 가 생기는 구간.

Watts–Strogatz 재배선 확률 p 를 올리면 평균 경로가 먼저 무너지고
뭉침은 나중에 무너진다. 그 사이가 small-world.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx

n, k, reps = 500, 10, 12
ps = np.logspace(-4, 0, 16)

G0 = nx.watts_strogatz_graph(n, k, 0.0, seed=0)
C0 = nx.average_clustering(G0)
L0 = nx.average_shortest_path_length(G0)

Cs, Ls = [], []
for p in ps:
    cc, ll = [], []
    for r in range(reps):
        G = nx.watts_strogatz_graph(n, k, p, seed=r)
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        cc.append(nx.average_clustering(G))
        ll.append(nx.average_shortest_path_length(G))
    Cs.append(np.mean(cc)); Ls.append(np.mean(ll))
Cs, Ls = np.array(Cs) / C0, np.array(Ls) / L0

plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
fig, ax = plt.subplots(figsize=(7.0, 3.7))
CC, LC = CAT[2], CAT[0]                      # 뭉침 · 평균 경로 (나란한 두 지표)
ax.axvspan(0.003, 0.08, color=FAINT, lw=0, zorder=0)
ax.text(0.0155, 1.1, "small-world 구간", ha="center", va="center", fontsize=11, color=TEXT2)
ax.semilogx(ps, Cs, "o-", color=CC, lw=2, ms=5, zorder=3)
ax.semilogx(ps, Ls, "s-", color=LC, lw=2, ms=5, zorder=3)
ax.text(0.11, 0.93, "뭉침  C(p) / C(0)", color=CC, fontsize=11.5, fontweight="bold", va="center")
ax.text(1.1e-4, 0.45, "평균 경로  L(p) / L(0)", color=LC, fontsize=11.5, fontweight="bold", va="center")

i = int(np.argmin(np.abs(ps - 0.004)))      # 본문이 짚는 점: p = 0.004
ax.plot([ps[i]] * 2, [Ls[i], Cs[i]], color=ACCENT_INK, lw=1.2, ls=(0, (3, 2)), zorder=2)
ax.plot([ps[i]] * 2, [Ls[i], Cs[i]], "o", ms=8, mfc="none", mec=ACCENT_INK, mew=1.8, zorder=4)
ax.annotate(f"p = {ps[i]:.3f}\n경로 {Ls[i]:.0%} · 뭉침 {Cs[i]:.0%}", xy=(ps[i], Ls[i]),
            xytext=(0.0095, 0.53), fontsize=11, color=ACCENT_INK, fontweight="bold", va="center",
            linespacing=1.4, arrowprops=dict(arrowstyle="-|>", color=ACCENT_INK, lw=1.2,
                                             connectionstyle="arc3,rad=0.25"))

ax.set_xlabel("재배선 확률  p", fontsize=11.5)
ax.set_ylabel("p = 0일 때 대비 비율", fontsize=11.5)
ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
save(fig, img("c04", "smallworld.png"))
for p, c, l in zip(ps, Cs, Ls):
    print(f"  p={p:.5f}  C/C0={c:.3f}  L/L0={l:.3f}")

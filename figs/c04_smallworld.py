"""04 §1 — small-world 가 생기는 구간.

Watts–Strogatz 재배선 확률 p 를 올리면 평균 경로가 먼저 무너지고
뭉침은 나중에 무너진다. 그 사이가 small-world.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx

OUT = outdir("c04")

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
Cs, Ls = np.array(Cs), np.array(Ls)

fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.semilogx(ps, Cs / C0, "o-", color=TEAL, lw=2, ms=5, label="뭉침  C(p) / C(0)")
ax.semilogx(ps, Ls / L0, "s-", color=ORANGE, lw=2, ms=5,
            label="평균 경로  L(p) / L(0)")
ax.axvspan(0.003, 0.08, color=PURPLE, alpha=0.12)
ax.text(0.015, 0.45, "small-world\n구간", ha="center", fontsize=11,
        color=PLUM, fontweight="bold")
ax.set_xlabel("재배선 확률  p")
ax.set_ylabel("p = 0 일 때 대비 비율")
ax.set_ylim(0, 1.05)
ax.set_title("경로는 먼저 짧아지고, 뭉침은 늦게까지 남는다", color=NAVY)
ax.legend(frameon=False, loc="lower left")

fig.savefig(os.path.join(OUT, "smallworld.png"))
print("wrote", os.path.join(OUT, "smallworld.png"))
for p, c, l in zip(ps, Cs / C0, Ls / L0):
    print(f"  p={p:.5f}  C/C0={c:.3f}  L/L0={l:.3f}")

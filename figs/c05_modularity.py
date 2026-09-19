"""05 §1 — 모듈성 Q 는 무엇을 재는가.

같은 그래프, 세 가지 분할. Q 가 어떻게 달라지는가.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx
from networkx.algorithms.community import modularity

OUT = outdir("c05")

n_comm, size, p_in, p_out = 4, 30, 0.22, 0.012
sizes = [size] * n_comm
P = np.full((n_comm, n_comm), p_out); np.fill_diagonal(P, p_in)
G = nx.stochastic_block_model(sizes, P, seed=3)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
pos = nx.spring_layout(G, seed=3, k=0.30)

truth = [set(i for i in G.nodes() if i // size == c) for c in range(n_comm)]
rng = np.random.default_rng(1)
rand_lab = rng.integers(0, n_comm, len(G))
random_part = [set(np.array(G.nodes())[rand_lab == c]) for c in range(n_comm)]
random_part = [s for s in random_part if s]
one_part = [set(G.nodes())]

palette = [TEAL, PURPLE, ORANGE, OLIVE, BLUE, RED]

cases = [("제대로 된 분할", truth),
         ("무작위 분할", random_part),
         ("전부 한 덩어리", one_part)]

fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.0), gridspec_kw={"wspace": 0.05})
for ax, (name, parts) in zip(axes, cases):
    cmap = {}
    for i, c in enumerate(parts):
        for v in c:
            cmap[v] = palette[i % len(palette)]
    Q = modularity(G, parts)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LGREY, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=[cmap[v] for v in G.nodes()],
                           linewidths=0, node_size=70)
    ax.set_title(name, color=NAVY, fontsize=12)
    ax.text(0.5, -0.03, f"Q = {Q:.3f}", transform=ax.transAxes, ha="center",
            va="top", fontsize=13, fontweight="bold",
            color=RED if Q < 0.1 else NAVY)
    ax.margins(0.06); ax.set_axis_off()
    print(f"{name:<14} 커뮤니티 {len(parts)}개  Q = {Q:.4f}")

fig.suptitle("Q = (커뮤니티 안의 엣지 비율) − (우연이라면 기대되는 비율)",
             color=NAVY, fontsize=13, fontweight="bold", y=1.02)
fig.savefig(os.path.join(OUT, "modularity.png"))
print("wrote", os.path.join(OUT, "modularity.png"))

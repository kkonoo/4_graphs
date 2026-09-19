"""07 §3 — 레이아웃은 데이터가 아니다.

완전히 같은 그래프, 완전히 같은 커뮤니티. 배치 알고리즘만 바꿨다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx
from networkx.algorithms.community import louvain_communities

OUT = outdir("c07")

n_comm, size, p_in, p_out = 4, 35, 0.16, 0.012
P = np.full((n_comm, n_comm), p_out); np.fill_diagonal(P, p_in)
G = nx.stochastic_block_model([size] * n_comm, P, seed=8)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

parts = louvain_communities(G, seed=0)
palette = [TEAL, PURPLE, ORANGE, OLIVE, BLUE, RED]
cmap = {}
for i, c in enumerate(parts):
    for v in c:
        cmap[v] = palette[i % len(palette)]
cols = [cmap[v] for v in G.nodes()]

layouts = [
    ("spring (seed=1)", nx.spring_layout(G, seed=1, k=0.30)),
    ("spring (seed=2)", nx.spring_layout(G, seed=2, k=0.30)),
    ("Kamada–Kawai", nx.kamada_kawai_layout(G)),
    ("circular", nx.circular_layout(G)),
]

fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.4), gridspec_kw={"wspace": 0.04})
for ax, (name, pos) in zip(axes, layouts):
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LGREY, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cols, linewidths=0,
                           node_size=55)
    ax.set_title(name, color=NAVY, fontsize=12)
    ax.margins(0.06); ax.set_axis_off()

fig.suptitle("같은 그래프, 같은 커뮤니티 — 노드 사이 거리를 해석하지 마세요",
             color=NAVY, fontsize=13, fontweight="bold", y=1.02)
fig.savefig(os.path.join(OUT, "layout.png"))
print("wrote", os.path.join(OUT, "layout.png"))
print(f"노드 {G.number_of_nodes()}, 엣지 {G.number_of_edges()}, "
      f"커뮤니티 {len(parts)}개 — 네 그림 모두 동일")

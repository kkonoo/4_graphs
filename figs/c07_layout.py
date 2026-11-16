"""07 §3 — 레이아웃은 데이터가 아니다.

완전히 같은 그래프, 완전히 같은 커뮤니티. 배치 알고리즘만 바꿨다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from networkx.algorithms.community import louvain_communities

n_comm, size, p_in, p_out = 4, 35, 0.16, 0.012
P = np.full((n_comm, n_comm), p_out); np.fill_diagonal(P, p_in)
G = nx.stochastic_block_model([size] * n_comm, P, seed=8)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

parts = louvain_communities(G, seed=0)
palette = list(CAT)                       # 커뮤니티 = 나란한 범주
if len(parts) > len(palette):
    palette += [shade(c, 0.3) for c in CAT]
cmap = {}
for i, c in enumerate(parts):
    for v in c:
        cmap[v] = palette[i % len(palette)]
cols = [cmap[v] for v in G.nodes()]

layouts = [
    ("① spring (seed = 1)", nx.spring_layout(G, seed=1, k=0.30)),
    ("② spring (seed = 2)", nx.spring_layout(G, seed=2, k=0.30)),
    ("③ Kamada–Kawai", nx.kamada_kawai_layout(G)),
    ("④ circular", nx.circular_layout(G)),
]

fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4),
                         gridspec_kw={"wspace": 0.06, "hspace": 0.14})
for ax, (name, pos) in zip(axes.ravel(), layouts):
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LINE, width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cols, linewidths=0, node_size=30)
    # 그리는 범위만 맞춤: 각 배치를 패널 가운데에, 같은 크기로
    xy = np.array([pos[v] for v in G.nodes()])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    c = (lo + hi) / 2
    h = (hi - lo).max() / 2 * 1.07
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(0.5, 1.01, name, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12.5, fontweight="bold", color=TEXT)

save(fig, img("c07", "layout.png"))
print(f"노드 {G.number_of_nodes()}, 엣지 {G.number_of_edges()}, "
      f"커뮤니티 {len(parts)}개 — 네 그림 모두 동일")

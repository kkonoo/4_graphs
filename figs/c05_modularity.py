"""05 §1 — 모듈성 Q 는 무엇을 재는가.

같은 그래프, 세 가지 분할. Q 가 어떻게 달라지는가.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from networkx.algorithms.community import modularity

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

NEUTRAL = "#9AA1AB"                     # 나눔이 없음 = 한 가지 무채색
cases = [("① 제대로 된 분할", truth, CAT),
         ("② 무작위 분할", random_part, CAT),
         ("③ 전부 한 덩어리", one_part, [NEUTRAL])]

fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.5), gridspec_kw={"wspace": 0.06})
for ax, (name, parts, pal) in zip(axes, cases):
    cmap = {v: pal[i % len(pal)] for i, c in enumerate(parts) for v in c}
    Q = modularity(G, parts)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LINE, width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[cmap[v] for v in G.nodes()],
                           linewidths=0, node_size=26)
    ax.margins(0.04); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(0.5, 1.02, name, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT)
    ax.text(0.5, -0.02, f"Q = {Q:.3f}".replace("-", "−"), transform=ax.transAxes,
            ha="center", va="top", fontsize=13, fontweight="bold", color=TEXT)
    print(f"{name:<14} 커뮤니티 {len(parts)}개  Q = {Q:.4f}")

save(fig, img("c05", "modularity.png"))

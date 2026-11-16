"""03 §2 — 뭉침 계수(clustering coefficient)의 정의.

같은 차수(k=5)인 노드 v. 이웃끼리 얼마나 이어져 있는가만 다르다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx


def star_with(extra_edges):
    G = nx.Graph()
    hub = "v"
    nbrs = [f"n{i}" for i in range(5)]
    G.add_edges_from((hub, x) for x in nbrs)
    G.add_edges_from((nbrs[a], nbrs[b]) for a, b in extra_edges)
    return G, hub, nbrs

cases = [([], "① 이웃끼리 전혀 안 이어짐"),
         ([(0, 1), (1, 2), (3, 4)], "② 일부만 이어짐"),
         ([(a, b) for a in range(5) for b in range(a + 1, 5)], "③ 이웃끼리 전부 이어짐")]

HUB, NBR, NBR_EDGE = PRIMARY, CAT[1], CAT[2]     # 나 · 이웃 · 이웃 사이 엣지 (세 패널 같은 뜻 = 같은 색)
HUB_EDGE = "#C9CED6"                            # v와 이웃을 잇는 엣지 (배경)

fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.7), gridspec_kw={"wspace": 0.08})

for ax, (extra, name) in zip(axes, cases):
    G, hub, nbrs = star_with(extra)
    C = nx.clustering(G, hub)
    k = G.degree(hub)
    possible = k * (k - 1) // 2
    actual = len(extra)

    pos = {hub: (0, 0)}
    for i, x in enumerate(nbrs):
        a = 2 * np.pi * i / 5 + np.pi / 2
        pos[x] = (np.cos(a), np.sin(a))

    nbr_edges = [(u, v) for u, v in G.edges() if hub not in (u, v)]
    hub_edges = [(u, v) for u, v in G.edges() if hub in (u, v)]
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=hub_edges, edge_color=HUB_EDGE, width=1.4)
    if nbr_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=nbr_edges, edge_color=NBR_EDGE, width=2.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[hub], node_color=HUB,
                           linewidths=0, node_size=520)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nbrs, node_color=NBR,
                           linewidths=0, node_size=230)
    ax.text(0, 0, "v", ha="center", va="center", fontsize=12, fontweight="bold",
            color=on(HUB), zorder=5)

    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.15, 1.25); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    ax.text(0.5, 1.03, name, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT)
    ax.text(0.5, -0.02, f"이웃 사이 엣지 {actual} / 가능한 {possible}", transform=ax.transAxes,
            ha="center", va="top", fontsize=11, color=TEXT2)
    ax.text(0.5, -0.13, f"C = {C:.1f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=shade(NBR_EDGE, 0.15) if actual else TEXT)
    print(f"{name:<16} k = {k}  이웃 사이 엣지 {actual} / 가능한 {possible}  C = {C:.1f}")

save(fig, img("c03", "clustering.png"))

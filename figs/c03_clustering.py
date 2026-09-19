"""03 §2 — 뭉침 계수(clustering coefficient)의 정의.

같은 차수(k=5)인 두 노드. 이웃끼리 얼마나 이어져 있는가만 다르다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx

OUT = outdir("c03")

def star_with(extra_edges):
    G = nx.Graph()
    hub = "v"
    nbrs = [f"n{i}" for i in range(5)]
    G.add_edges_from((hub, x) for x in nbrs)
    G.add_edges_from((nbrs[a], nbrs[b]) for a, b in extra_edges)
    return G, hub, nbrs

cases = [([], "이웃끼리 전혀 안 이어짐"),
         ([(0, 1), (1, 2), (3, 4)], "일부만 이어짐"),
         ([(a, b) for a in range(5) for b in range(a + 1, 5)], "이웃끼리 전부 이어짐")]

fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), gridspec_kw={"wspace": 0.06})

for ax, (extra, label) in zip(axes, cases):
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
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=hub_edges,
                           edge_color="#C3CED6", width=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=nbr_edges,
                           edge_color=RED, width=2.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[hub], node_color=NAVY,
                           linewidths=0, node_size=620)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nbrs, node_color=TEAL,
                           linewidths=0, node_size=420)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="white",
                            font_weight="bold")
    ax.set_title(label, color=NAVY, fontsize=12)
    ax.text(0.5, -0.06,
            f"이웃 사이 엣지 {actual} / 가능한 {possible}   →   C = {C:.1f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            color=RED if actual else "#4A4A4A")
    ax.margins(0.22); ax.set_axis_off()

fig.suptitle("차수는 셋 다 k = 5. 뭉침 계수만 다르다.", color=NAVY,
             fontsize=13, fontweight="bold", y=1.02)
fig.savefig(os.path.join(OUT, "clustering.png"))
print("wrote", os.path.join(OUT, "clustering.png"))

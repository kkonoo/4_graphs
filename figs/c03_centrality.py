"""03 §3 — 중심성 네 가지, 같은 그래프.

노드 크기 = 그 중심성. 1위가 지표마다 옮겨 다닌다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx

OUT = outdir("c03")

G = nx.karate_club_graph()
pos = nx.kamada_kawai_layout(G)

measures = [
    ("차수 (degree)", nx.degree_centrality(G), TEAL),
    ("매개 (betweenness)", nx.betweenness_centrality(G), ORANGE),
    ("근접 (closeness)", nx.closeness_centrality(G), PURPLE),
    ("고유벡터 (eigenvector)", nx.eigenvector_centrality(G), BLUE),
]

fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.6), gridspec_kw={"wspace": 0.05})

for ax, (name, c, col) in zip(axes, measures):
    v = np.array([c[n] for n in G.nodes()])
    # 지표마다 값의 범위가 달라서(특히 근접 중심성) 최소–최대로 정규화해야
    # 차이가 눈에 보입니다. 크기는 순위를 보여주기 위한 것입니다.
    rel = (v - v.min()) / (v.max() - v.min())
    sizes = 40 + 880 * rel
    top3 = sorted(c, key=c.get, reverse=True)[:3]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LGREY, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=col, linewidths=0,
                           node_size=sizes, alpha=0.92)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=top3, node_color=RED,
                           linewidths=0, node_size=[sizes[i] for i in top3])
    nx.draw_networkx_labels(G, pos, ax=ax, labels={i: str(i) for i in top3},
                            font_size=9, font_color="white", font_weight="bold")
    ax.set_title(name, color=NAVY, fontsize=12)
    ax.text(0.5, -0.04, "상위 3개: " + ", ".join(str(i) for i in top3),
            transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
            color=RED)
    ax.margins(0.10); ax.set_axis_off()

fig.suptitle("같은 네트워크, 네 가지 중심성 — 1위가 달라진다", color=NAVY,
             fontsize=13, fontweight="bold", y=1.03)
fig.savefig(os.path.join(OUT, "centrality.png"))
print("wrote", os.path.join(OUT, "centrality.png"))
for name, c, _ in measures:
    print(f"  {name}: {sorted(c, key=c.get, reverse=True)[:5]}")
pr = nx.pagerank(G)
print("  PageRank:", sorted(pr, key=pr.get, reverse=True)[:5])
print(f"  평균 경로 {nx.average_shortest_path_length(G):.3f}, "
      f"지름 {nx.diameter(G)}, 평균 C {nx.average_clustering(G):.3f}, "
      f"transitivity {nx.transitivity(G):.3f}")

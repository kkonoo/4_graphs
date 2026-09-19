"""04 §1 — 세 가지 생성 모델.

노드 수와 엣지 수를 맞춘 ER / WS / BA. 무엇을 재현하고 무엇을 못 하는가.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx

OUT = outdir("c04")

n, k = 120, 4
ER = nx.gnm_random_graph(n, n * k // 2, seed=2)
WS = nx.watts_strogatz_graph(n, k, 0.08, seed=2)
BA = nx.barabasi_albert_graph(n, k // 2, seed=2)

models = [("Erdős–Rényi (ER)\n무작위로 엣지를 뿌림", ER, BLUE),
          ("Watts–Strogatz (WS)\n고리에서 일부만 재배선", WS, TEAL),
          ("Barabási–Albert (BA)\n인기 있는 노드에 더 붙음", BA, ORANGE)]

fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.2), gridspec_kw={"wspace": 0.06})

for ax, (name, G, col) in zip(axes, models):
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    pos = (nx.circular_layout(G) if name.startswith("Watts")
           else nx.spring_layout(G, seed=3, k=0.35))
    deg = np.array([d for _, d in G.degree()])
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LGREY, width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=col, linewidths=0,
                           node_size=14 + 7 * deg)
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)
    ax.set_title(name, color=NAVY, fontsize=12)
    ax.text(0.5, -0.03,
            f"뭉침 C = {C:.3f}   ·   평균 경로 = {L:.2f}   ·   최대 차수 = {deg.max()}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
            color="#4A4A4A")
    ax.margins(0.08); ax.set_axis_off()
    print(f"{name.splitlines()[0]:<28} C={C:.4f}  L={L:.3f}  kmax={deg.max()}")

fig.suptitle(f"노드 {n}개 · 평균 차수 {k} 로 맞춘 세 모델", color=NAVY,
             fontsize=13, fontweight="bold", y=1.02)
fig.savefig(os.path.join(OUT, "models.png"))
print("wrote", os.path.join(OUT, "models.png"))

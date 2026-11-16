"""04 §1 — 세 가지 생성 모델.

노드 수와 엣지 수를 맞춘 ER / WS / BA. 무엇을 재현하고 무엇을 못 하는가.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx

n, k = 120, 4
ER = nx.gnm_random_graph(n, n * k // 2, seed=2)
WS = nx.watts_strogatz_graph(n, k, 0.08, seed=2)
BA = nx.barabasi_albert_graph(n, k // 2, seed=2)

# 나란한 세 모델 → CAT[0..2]; 본문 표에서 굵게 쓴 값(WS의 C, BA의 경로·최대 차수)만 진하게
models = [("Erdős–Rényi (ER)", "무작위로 엣지를 뿌림", ER, CAT[0], set()),
          ("Watts–Strogatz (WS)", "고리에서 일부만 재배선", WS, CAT[1], {"C"}),
          ("Barabási–Albert (BA)", "인기 있는 노드에 더 붙음", BA, CAT[2], {"L", "K"})]
marks = ["①", "②", "③"]

plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
fig, axes = plt.subplots(1, 3, figsize=(8.4, 4.1), gridspec_kw={"wspace": 0.08})
fig.subplots_adjust(top=0.84, bottom=0.22, left=0.01, right=0.99)

for ax, mk, (name, sub, G, col, bold) in zip(axes, marks, models):
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    pos = (nx.circular_layout(G) if name.startswith("Watts")
           else nx.spring_layout(G, seed=3, k=0.35))
    deg = np.array([d for _, d in G.degree()])
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LINE, width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=col, linewidths=0,
                           node_size=6 + 4 * deg)
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)
    ax.margins(0.06); ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    ax.text(0.5, 1.12, f"{mk} {name}", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT)
    ax.text(0.5, 1.03, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, color=TEXT2)
    rows = [("C", f"뭉침 C = {C:.3f}"), ("L", f"평균 경로 = {L:.2f}"),
            ("K", f"최대 차수 = {deg.max()}")]
    for j, (key, txt) in enumerate(rows):
        b = key in bold
        ax.text(0.5, -0.03 - 0.095 * j, txt, transform=ax.transAxes, ha="center", va="top",
                fontsize=11.5 if b else 11, fontweight="bold" if b else "regular",
                color=TEXT if b else TEXT2)
    print(f"{name:<28} C={C:.4f}  L={L:.3f}  kmax={deg.max()}")

save(fig, img("c04", "models.png"))

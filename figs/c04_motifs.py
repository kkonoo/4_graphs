"""04 §3 — 3-노드 모티프와 그 유의성.

왼쪽: 방향 그래프에서 자주 이야기되는 3-노드 부분구조.
오른쪽: 심어둔 feed-forward loop 가 방향 보존 재배선 null 대비 유의한가.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx
from itertools import combinations

OUT = outdir("c04")

# ── 모티프 도식 ───────────────────────────────────────────────────────
shapes = [
    ("Cascade\n(연쇄)", [("A", "B"), ("B", "C")]),
    ("Feed-forward loop\n(FFL)", [("A", "B"), ("B", "C"), ("A", "C")]),
    ("Feedback loop\n(되먹임)", [("A", "B"), ("B", "C"), ("C", "A")]),
    ("Fan-out\n(공동 조절)", [("A", "B"), ("A", "C")]),
]

# ── FFL 을 심은 합성 조절 네트워크 ────────────────────────────────────
rng = np.random.default_rng(7)
n_genes, n_ffl, n_random = 80, 25, 160
D = nx.DiGraph()
D.add_nodes_from(range(n_genes))
for _ in range(n_ffl):                      # FFL 을 의도적으로 심는다
    a, b, c = rng.choice(n_genes, 3, replace=False)
    D.add_edges_from([(int(a), int(b)), (int(b), int(c)), (int(a), int(c))])
while D.number_of_edges() < n_ffl * 3 + n_random:   # 나머지는 무작위 엣지
    u, v = rng.integers(0, n_genes, 2)
    if u != v:
        D.add_edge(int(u), int(v))


def count_ffl(G):
    """A→B, B→C, A→C 를 모두 갖춘 순서쌍의 수."""
    c = 0
    for b in G.nodes():
        for a in G.predecessors(b):
            for d in G.successors(b):
                if d != a and G.has_edge(a, d):
                    c += 1
    return c


obs = count_ffl(D)
N = 500
null = []
for i in range(N):
    H = D.copy()
    # in/out 차수를 모두 보존한 채 방향 엣지를 섞는다
    try:
        nx.directed_edge_swap(H, nswap=5 * H.number_of_edges(),
                              max_tries=200 * H.number_of_edges(),
                              seed=int(rng.integers(1e9)))
    except nx.NetworkXAlgorithmError:
        pass
    null.append(count_ffl(H))
null = np.array(null)
z = (obs - null.mean()) / null.std(ddof=1)
p = (np.sum(null >= obs) + 1) / (N + 1)

fig = plt.figure(figsize=(15.0, 4.6))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 2.1], wspace=0.30)

for i, (name, edges) in enumerate(shapes):
    ax = fig.add_subplot(gs[0, i])
    g = nx.DiGraph(edges)
    pos = {"A": (0, 1), "B": (-0.9, -0.6), "C": (0.9, -0.6)}
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#7C93A3", width=2.0,
                           arrowsize=16, node_size=760,
                           connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=NAVY, linewidths=0,
                           node_size=760)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=11, font_color="white",
                            font_weight="bold")
    ax.set_title(name, color=NAVY, fontsize=11.5)
    ax.margins(0.28); ax.set_axis_off()

ax = fig.add_subplot(gs[0, 4])
ax.hist(null, bins=24, color=GREY, alpha=0.85, label=f"null (재배선 {N}회)")
ax.axvline(obs, color=RED, lw=2.6, label="관측값")
ax.axvline(null.mean(), color=NAVY, lw=1.4, ls="--", label="null 평균")
ax.set_xlabel("FFL 개수"); ax.set_ylabel("빈도")
ax.set_title(f"FFL 이 유의하게 많은가 — Z = {z:.1f}", color=NAVY, fontsize=12)
ax.legend(frameon=False, fontsize=9.5)

fig.savefig(os.path.join(OUT, "motifs.png"))
print("wrote", os.path.join(OUT, "motifs.png"))
print(f"노드 {D.number_of_nodes()}, 엣지 {D.number_of_edges()}")
print(f"관측 FFL {obs} | null 평균 {null.mean():.1f} ± {null.std(ddof=1):.1f} "
      f"| Z = {z:.2f} | p = {p:.4f}")

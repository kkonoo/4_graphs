"""02 §1 — 이분 그래프와 사영(projection).

유전자–경로 이분 그래프를 유전자–유전자로 접으면, 큰 경로 하나가 거대한
clique 을 만든다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from itertools import combinations

genes = [f"G{i}" for i in range(1, 11)]
# P3 는 유전자 6개가 붙어 있는 '큰 경로'
pathways = {"P1": ["G1", "G2"], "P2": ["G2", "G3", "G4"],
            "P3": ["G4", "G5", "G6", "G7", "G8", "G9"], "P4": ["G9", "G10"]}

B = nx.Graph()
B.add_nodes_from(genes, bipartite=0)
B.add_nodes_from(pathways, bipartite=1)
for p, gs_ in pathways.items():
    B.add_edges_from((g, p) for g in gs_)

P = nx.bipartite.weighted_projected_graph(B, genes)

big = set(pathways["P3"])
posp = nx.spring_layout(P, seed=9, k=0.9)
n_clique = len(list(combinations(big, 2)))

# ── colours ──────────────────────────────────────────────────────────────
GENE, BIGG, PATH = CAT[2], CAT[1], PRIMARY   # 유전자 · P3 소속 유전자 · 경로
EDGE = shade(LINE, 0.08)
P3EDGE = mix(BIGG, 0.55)
CLIQ = ACCENT_INK                              # 강조색은 이 한 곳: P3 clique 엣지

fig = plt.figure(figsize=(8.0, 4.4))
axL = fig.add_axes([0.0, 0.09, 0.40, 0.82])
axR = fig.add_axes([0.45, 0.09, 0.55, 0.82])

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def names(ax, pos, nodes):
    for v in nodes:
        x, y = pos[v]
        ax.text(x, y, v, ha="center", va="center", fontsize=11, fontweight="bold",
                color="white", zorder=5)

# ── ① 이분 그래프 ─────────────────────────────────────────────────────────
ax = axL
pos = {}
for i, g in enumerate(genes):
    pos[g] = (0, -i * 0.9)
for i, p in enumerate(pathways):
    pos[p] = (2.6, -4.05 + (1.5 - i) * 2.4)
other_e = [(u, v) for u, v in B.edges() if "P3" not in (u, v)]
p3_e = [(u, v) for u, v in B.edges() if "P3" in (u, v)]
nx.draw_networkx_edges(B, pos, ax=ax, edgelist=other_e, edge_color=EDGE, width=1.5)
nx.draw_networkx_edges(B, pos, ax=ax, edgelist=p3_e, edge_color=P3EDGE, width=1.7)
nx.draw_networkx_nodes(B, pos, ax=ax, nodelist=genes,
                       node_color=[BIGG if g in big else GENE for g in genes],
                       linewidths=0, node_size=560)
nx.draw_networkx_nodes(B, pos, ax=ax, nodelist=list(pathways), node_color=PATH,
                       node_shape="s", linewidths=0, node_size=760)
names(ax, pos, list(B.nodes()))
ax.set_xlim(-0.5, 3.1); ax.set_ylim(-8.55, 0.45)
clean(ax)
ax.text(0.5, 1.03, "① 이분 그래프 · 유전자 10개 · 경로 4개", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=12, fontweight="bold", color=TEXT)
ax.text(0.5, -0.03, f"엣지 {B.number_of_edges()}개", transform=ax.transAxes,
        ha="center", va="top", fontsize=12, fontweight="bold", color=TEXT)

# ── ② 사영 ───────────────────────────────────────────────────────────────
ax = axR
cl_e = [(u, v) for u, v in P.edges() if (u in big and v in big)]
ot_e = [(u, v) for u, v in P.edges() if not (u in big and v in big)]
nx.draw_networkx_edges(P, posp, ax=ax, edgelist=ot_e, edge_color=EDGE, width=1.5)
nx.draw_networkx_edges(P, posp, ax=ax, edgelist=cl_e, edge_color=CLIQ, width=2.2)
nx.draw_networkx_nodes(P, posp, ax=ax,
                       node_color=[BIGG if g in big else GENE for g in P.nodes()],
                       linewidths=0, node_size=560)
names(ax, posp, list(P.nodes()))
ax.margins(0.10)
clean(ax)
ax.text(0.5, 1.03, "② 사영 · 유전자–유전자", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=12, fontweight="bold", color=TEXT)
ax.text(0.5, -0.03,
        f"엣지 {P.number_of_edges()}개 중 {n_clique}개가 P3 하나에서 나온 clique",
        transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold",
        color=TEXT)

save(fig, img("c02", "projection.png"))
print("  bipartite edges:", B.number_of_edges(),
      "| projected edges:", P.number_of_edges(),
      "| P3 clique edges:", n_clique)

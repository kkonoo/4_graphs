"""02 §3 — 연결성: component, bridge, articulation node, 그리고 방향 그래프의 SCC.

왼쪽: 무방향 그래프의 연결 요소 · bridge · 절단점.
오른쪽: 방향 그래프 — 방향을 지우면 한 덩어리지만 SCC는 3개.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx

NEUTRAL = "#9AA1AB"
NODE_BG = shade(NEUTRAL, 0.15)      # 평범한 노드 = 무채색 (흰 번호가 읽히도록 조금 어둡게)
KEY = PRIMARY                        # 지우면 그래프가 쪼개지는 것 (bridge · 절단점)
EDGE = "#C5CAD2"                     # 노드 11개짜리 작은 그래프라 LINE보다 한 단계 진하게
NS = 480

# ── 왼쪽: 무방향 그래프 ───────────────────────────────────────────────
G = nx.Graph()
G.add_edges_from([
    # 덩어리 A
    (0, 1), (0, 2), (1, 2), (2, 3), (1, 3),
    # 다리
    (3, 4),
    # 덩어리 B
    (4, 5), (4, 6), (5, 6), (6, 7), (5, 7),
    # 떨어져 있는 작은 덩어리
    (8, 9), (9, 10), (8, 10),
])
bridges = list(nx.bridges(G))
arts = list(nx.articulation_points(G))
comps = list(nx.connected_components(G))

pos = nx.spring_layout(G, seed=12, k=0.85)
# 손으로 정한 배치 (spring 배치에서는 노드가 겹쳐 보임) — 구조는 같음
pos = {0: (0.0, 1.0), 1: (0.75, 0.35), 2: (0.75, 1.65), 3: (1.55, 1.0),
       4: (2.75, 1.0), 5: (3.55, 1.65), 6: (3.55, 0.35), 7: (4.3, 1.0),
       8: (1.55, -1.25), 9: (2.15, -0.6), 10: (2.75, -1.25)}

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), gridspec_kw={"wspace": 0.08})

def clean(ax):
    ax.margins(0.1); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

ax = axes[0]
is_br = [(u, v) in bridges or (v, u) in bridges for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=[KEY if b else EDGE for b in is_br],
                       width=[2.8 if b else 1.4 for b in is_br])
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[KEY if n in arts else NODE_BG for n in G.nodes()],
                       linewidths=0, node_size=NS)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_color="white", font_weight="bold",
                        font_family="Pretendard")
clean(ax)
ax.set_aspect("equal", adjustable="datalim")

# 직접 라벨
u, v = bridges[0]
m = (np.array(pos[u]) + np.array(pos[v])) / 2
ax.annotate("bridge", xy=m, xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color=KEY)
for a in arts:
    ax.annotate("절단점", xy=pos[a], xytext=(0, -15), textcoords="offset points", ha="center",
                va="top", fontsize=11, fontweight="bold", color=KEY)

ax.text(0.5, 1.03, "① 무방향 그래프", transform=ax.transAxes, ha="center", va="bottom",
        fontsize=12.5, fontweight="bold", color=TEXT)
ax.text(0.5, -0.03, f"연결 요소 {len(comps)}개 · bridge {len(bridges)}개 · 절단점 {len(arts)}개",
        transform=ax.transAxes, ha="center", va="top", fontsize=11, color=TEXT2)

# ── 오른쪽: 방향 그래프의 SCC ─────────────────────────────────────────
D = nx.DiGraph()
D.add_edges_from([
    (0, 1), (1, 2), (2, 0),          # SCC 1
    (2, 3),
    (3, 4), (4, 5), (5, 3),          # SCC 2
    (5, 6),
    (6, 7), (7, 6),                  # SCC 3
])
sccs = sorted(nx.strongly_connected_components(D), key=lambda s: min(s))
wcc = nx.number_weakly_connected_components(D)
palette = CAT[:3]
cmap = {n: palette[i % len(palette)] for i, s in enumerate(sccs) for n in s}

ax = axes[1]
posd = nx.kamada_kawai_layout(D)
pair = [(6, 7), (7, 6)]
rest = [e for e in D.edges() if e not in pair]
nx.draw_networkx_edges(D, posd, ax=ax, edgelist=rest, edge_color=NEUTRAL, width=1.4, arrowsize=14,
                       arrowstyle="-|>", node_size=NS, connectionstyle="arc3,rad=0.12")
nx.draw_networkx_edges(D, posd, ax=ax, edgelist=pair, edge_color=NEUTRAL, width=1.4, arrowsize=13,
                       arrowstyle="-|>", node_size=NS, connectionstyle="arc3,rad=0.45")
nx.draw_networkx_nodes(D, posd, ax=ax, node_color=[cmap[n] for n in D.nodes()],
                       linewidths=0, node_size=NS)
nx.draw_networkx_labels(D, posd, ax=ax, font_size=11, font_color="white", font_weight="bold",
                        font_family="Pretendard")
clean(ax)
ax.text(0.5, 1.03, "② 방향 그래프", transform=ax.transAxes, ha="center", va="bottom",
        fontsize=12.5, fontweight="bold", color=TEXT)
ax.text(0.5, -0.03, f"같은 색 = 강하게 연결된 요소(SCC) {len(sccs)}개\n방향을 지우면 전부 한 덩어리",
        transform=ax.transAxes, ha="center", va="top", fontsize=11, color=TEXT2, linespacing=1.45)

save(fig, img("c02", "connectivity.png"))
print("  components:", [sorted(c) for c in comps])
print("  bridges:", bridges, " articulation:", sorted(arts))
print("  SCCs:", [sorted(s) for s in sccs], " WCCs:", wcc)

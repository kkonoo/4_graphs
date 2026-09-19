"""05 §3 — resolution 파라미터와 resolution limit.

위: resolution 을 올리면 커뮤니티가 잘게 쪼개진다 (scanpy 의 그 파라미터).
아래: 모듈성 최대화가 '정답'보다 높은 점수를 주는 경우 — resolution limit.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
import igraph as ig
import leidenalg as la

OUT = outdir("c05")
palette = [TEAL, PURPLE, ORANGE, OLIVE, BLUE, RED, "#7FA8D0", "#C9A227",
           "#8FBC8F", "#D08770"]

# ── 위: resolution 훑기 ───────────────────────────────────────────────
n_comm, size, p_in, p_out = 6, 60, 0.12, 0.012
P = np.full((n_comm, n_comm), p_out); np.fill_diagonal(P, p_in)
G = nx.stochastic_block_model([size] * n_comm, P, seed=2)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
g = ig.Graph.from_networkx(G)
pos = nx.spring_layout(G, seed=2, k=0.24)

resolutions = np.round(np.arange(0.1, 4.01, 0.1), 2)
ks = []
for r in resolutions:
    p = la.find_partition(g, la.RBConfigurationVertexPartition,
                          resolution_parameter=float(r), seed=0, n_iterations=2)
    ks.append(len(p))

show = [0.2, 1.0, 3.0]
parts_show = []
for r in show:
    p = la.find_partition(g, la.RBConfigurationVertexPartition,
                          resolution_parameter=r, seed=0, n_iterations=2)
    parts_show.append((r, np.array(p.membership), len(p)))
    print(f"resolution {r}: 커뮤니티 {len(p)}개")

# ── 아래: resolution limit — 클리크 고리 ──────────────────────────────
def ring_of_cliques(n_cliques, k=5):
    R = nx.Graph()
    for i in range(n_cliques):
        nodes = [f"{i}_{j}" for j in range(k)]
        R.add_edges_from((a, b) for ii, a in enumerate(nodes) for b in nodes[ii+1:])
    for i in range(n_cliques):
        R.add_edge(f"{i}_0", f"{(i+1) % n_cliques}_0")
    return R

rows = []
for nc in (12, 24, 40):
    R = ring_of_cliques(nc)
    truth = [set(f"{i}_{j}" for j in range(5)) for i in range(nc)]
    found = louvain_communities(R, seed=0)
    rows.append((nc, modularity(R, truth), len(found), modularity(R, found)))
    print(f"클리크 {nc}개: 정답 Q={rows[-1][1]:.4f} | "
          f"찾은 답 {len(found)}개 Q={rows[-1][3]:.4f}")

# ── 그림 ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15.2, 8.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.38, wspace=0.22)

ax = fig.add_subplot(gs[0, 0])
ax.plot(resolutions, ks, "-", color=NAVY, lw=2)
for r, col in zip(show, [ORANGE, TEAL, PURPLE]):
    ax.axvline(r, color=col, ls="--", lw=1.4)
ax.axhline(n_comm, color=RED, ls=":", lw=1.5)
ax.text(3.4, n_comm + 1.2, "정답 6개", color=RED, fontsize=10)
ax.set_xlabel("resolution"); ax.set_ylabel("커뮤니티 수")
ax.set_title("resolution 이 클러스터 수를 정한다", color=NAVY, fontsize=12)

for i, (r, memb, k) in enumerate(parts_show):
    ax = fig.add_subplot(gs[0, i + 1])
    cols = [palette[m % len(palette)] for m in memb]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LGREY, width=0.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cols, linewidths=0,
                           node_size=42)
    ax.set_title(f"resolution = {r}  →  {k}개", color=NAVY, fontsize=12)
    ax.margins(0.05); ax.set_axis_off()

ax = fig.add_subplot(gs[1, :2])
nc_list = [r[0] for r in rows]
xx = np.arange(len(rows)); w = 0.36
ax.bar(xx - w/2, [r[1] for r in rows], w, color=TEAL,
       label="정답 분할의 Q (클리크마다 하나씩)")
ax.bar(xx + w/2, [r[3] for r in rows], w, color=RED, alpha=0.85,
       label="모듈성이 고른 분할의 Q")
for i, r in enumerate(rows):
    ax.text(i + w/2, r[3] + 0.006, f"{r[2]}개", ha="center", fontsize=10,
            color=RED, fontweight="bold")
    ax.text(i - w/2, r[1] + 0.006, f"{r[0]}개", ha="center", fontsize=10,
            color=TEAL, fontweight="bold")
ax.set_xticks(xx); ax.set_xticklabels([f"클리크 {n}개" for n in nc_list])
ax.set_ylim(0.75, 0.95); ax.set_ylabel("모듈성 Q")
ax.set_title("⚠ resolution limit — 네트워크가 커지면 '정답'보다 합친 답의 Q 가 더 높아진다",
             color=NAVY, fontsize=12)
ax.legend(frameon=False, fontsize=10, loc="upper left")

ax = fig.add_subplot(gs[1, 2:])
R = ring_of_cliques(24)
found = louvain_communities(R, seed=0)
cmap = {}
for i, c in enumerate(found):
    for v in c:
        cmap[v] = palette[i % len(palette)]
# 클리크마다 작은 덩어리로 배치해야 '몇 개를 하나로 묶었는지'가 보입니다
n_ring = 24
rpos = {}
for i in range(n_ring):
    a = 2 * np.pi * i / n_ring
    cx, cy = np.cos(a), np.sin(a)
    for j in range(5):
        b = 2 * np.pi * j / 5
        rpos[f"{i}_{j}"] = (cx + 0.085 * np.cos(b), cy + 0.085 * np.sin(b))
nx.draw_networkx_edges(R, rpos, ax=ax, edge_color="#D8DEE3", width=0.6)
nx.draw_networkx_nodes(R, rpos, ax=ax, node_color=[cmap[v] for v in R.nodes()],
                       linewidths=0, node_size=34)
ax.set_title(f"클리크 24개 고리 — 모듈성은 {len(found)}개로 묶는다 (같은 색 = 같은 커뮤니티)",
             color=NAVY, fontsize=12)
ax.margins(0.04); ax.set_axis_off()

fig.savefig(os.path.join(OUT, "resolution.png"))
print("wrote", os.path.join(OUT, "resolution.png"))

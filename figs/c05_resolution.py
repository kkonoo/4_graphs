"""05 §3 — resolution 파라미터와 resolution limit.

위: resolution 을 올리면 커뮤니티가 잘게 쪼개진다 (scanpy 의 그 파라미터).
아래: 모듈성 최대화가 '정답'보다 높은 점수를 주는 경우 — resolution limit.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
import igraph as ig
import leidenalg as la

NEUTRAL = "#9AA1AB"
# 커뮤니티가 많을 때: CAT 에 옅게·짙게 한 색을 더함 (모두 단색)
PAL = list(CAT[:4]) + [mix(CAT[1], 0.5), mix(CAT[2], 0.5), shade(CAT[1], 0.35),
                       shade(CAT[2], 0.4), mix(CAT[0], 0.8)]

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
plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
fig = plt.figure(figsize=(8.4, 6.4))
outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.55)
top = outer[0].subgridspec(1, 4, width_ratios=[1.45, 1, 1, 1], wspace=0.12)
bot = outer[1].subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.12)

def marker(ax, text, ha="left", x=0.0):
    ax.text(x, 1.05, text, transform=ax.transAxes, ha=ha, va="bottom",
            fontsize=12, fontweight="bold", color=TEXT)

def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

# ① resolution → 커뮤니티 수
ax = fig.add_subplot(top[0])
ax.axhline(n_comm, color=TEXT2, ls=(0, (2, 2)), lw=1.2, zorder=1)
ax.text(4.05, n_comm + 1.2, "정답 6개", color=TEXT2, fontsize=11, ha="right", va="bottom")
ax.plot(resolutions, ks, "-", color=PRIMARY, lw=2, zorder=3)
ks_show = [k for _, _, k in parts_show]
ax.plot(show, ks_show, "o", ms=6.5, mfc="white", mec=PRIMARY, mew=1.7, zorder=4)
ax.text(0.13, 3.0, f"{ks_show[0]}개", ha="center", va="bottom", fontsize=11, color=TEXT2,
        zorder=5, bbox=dict(fc="white", ec="none", pad=0.6))
ax.text(1.0, 10, f"{ks_show[1]}개", ha="center", va="bottom", fontsize=11, color=TEXT2)
ax.text(2.75, 30, f"{ks_show[2]}개", ha="right", va="bottom", fontsize=11, color=TEXT2)
ax.set_xlim(-0.2, 4.15); ax.set_ylim(0, 42)
ax.set_xticks([0, 1, 2, 3, 4]); ax.set_yticks([0, 10, 20, 30, 40])
ax.set_xlabel("resolution", fontsize=11.5)
ax.set_ylabel("커뮤니티 수", fontsize=11.5)
ax.grid(axis="y", color=LINE, lw=0.6); ax.set_axisbelow(True)
marker(ax, "① resolution과 커뮤니티 수", x=-0.25)

# 세 resolution에서의 분할
for i, (r, memb, k) in enumerate(parts_show):
    ax = fig.add_subplot(top[i + 1])
    if k == 1:
        cols = [NEUTRAL] * len(memb)
    else:
        cols = [PAL[m % len(PAL)] for m in memb]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LINE, width=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cols, linewidths=0, node_size=8)
    ax.margins(0.03); bare(ax)
    marker(ax, f"{r} → {k}개", ha="center", x=0.5)

# ② 정답 Q vs 모듈성이 고른 분할의 Q
ax = fig.add_subplot(bot[0])
C_T, C_F = CAT[2], CAT[0]                 # 정답 · 모듈성이 고른 분할 (나란한 두 막대)
nc_list = [r[0] for r in rows]
xx = np.arange(len(rows)); w = 0.36
ax.bar(xx - w/2, [r[1] for r in rows], w, color=C_T, label="정답 (클리크마다 하나)")
ax.bar(xx + w/2, [r[3] for r in rows], w, color=C_F, label="모듈성이 고른 분할")
for i, r in enumerate(rows):
    ax.text(i - w/2, r[1] + 0.003, f"{r[0]}개", ha="center", va="bottom", fontsize=11,
            color=C_T, fontweight="bold")
    ax.text(i + w/2, r[3] + 0.003, f"{r[2]}개", ha="center", va="bottom", fontsize=11,
            color=C_F, fontweight="bold")
ax.text(2 + w/2, 0.926, "정답보다\nQ가 높다", ha="center", va="bottom", fontsize=11,
        color=ACCENT_INK, fontweight="bold", linespacing=1.1)
ax.set_xticks(xx); ax.set_xticklabels([f"클리크 {n}개" for n in nc_list])
ax.set_xlim(-0.55, 2.6)
ax.set_ylim(0.75, 0.965); ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
ax.set_ylabel("모듈성 Q", fontsize=11.5)
ax.grid(axis="y", color=LINE, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=11, loc="upper left", handlelength=1.2,
          borderaxespad=0.1)
marker(ax, "② 클리크 고리: 정답과 모듈성의 Q", x=-0.2)

# ③ 클리크 24개 고리
ax = fig.add_subplot(bot[1])
R = ring_of_cliques(24)
found = louvain_communities(R, seed=0)
n_ring = 24
# 고리를 따라 커뮤니티를 정렬해, 이웃한 커뮤니티가 서로 다른 색을 갖게 함
def circ_angle(c):
    a = np.array([2 * np.pi * int(v.split("_")[0]) / n_ring for v in c])
    return np.arctan2(np.sin(a).mean(), np.cos(a).mean()) % (2 * np.pi)
order = sorted(found, key=circ_angle)
ring_pal = list(CAT[:4])
ring_cols = [ring_pal[i % 4] for i in range(len(order))]
if len(order) > 1 and ring_cols[-1] == ring_cols[0]:
    ring_cols[-1] = next(c for c in ring_pal if c not in (ring_cols[-2], ring_cols[0]))
cmap = {v: ring_cols[i] for i, c in enumerate(order) for v in c}
rpos = {}
for i in range(n_ring):
    a = 2 * np.pi * i / n_ring
    cx, cy = np.cos(a), np.sin(a)
    for j in range(5):
        b = 2 * np.pi * j / 5
        rpos[f"{i}_{j}"] = (cx + 0.085 * np.cos(b), cy + 0.085 * np.sin(b))
nx.draw_networkx_edges(R, rpos, ax=ax, edge_color="#C9CED6", width=0.6)
nx.draw_networkx_nodes(R, rpos, ax=ax, node_color=[cmap[v] for v in R.nodes()],
                       linewidths=0, node_size=14)
ax.text(0, 0.12, f"클리크 {n_ring}개 →\n커뮤니티 {len(found)}개", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=TEXT, linespacing=1.4)
ax.text(0, -0.32, "같은 색 = 같은 커뮤니티", ha="center", va="center", fontsize=11, color=TEXT2)
ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
bare(ax)
marker(ax, "③ 모듈성이 묶은 결과 (클리크 24개)", ha="center", x=0.5)

save(fig, img("c05", "resolution.png"))

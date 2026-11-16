"""06 §2 — 재시작 랜덤워크(RWR)와 restart 파라미터.

위 세 패널: 시드에서 출발한 확률이 네트워크로 퍼지는 모습 (세 패널 같은 색 눈금).
아래: alpha 를 1 에 가깝게 두면 결국 '차수 줄세우기'가 된다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from scipy.stats import spearmanr
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# ── 계산 (이전 스크립트와 동일) ───────────────────────────────────────────
n_comm, size, p_in, p_out = 5, 60, 0.12, 0.008
P = np.full((n_comm, n_comm), p_out); np.fill_diagonal(P, p_in)
G = nx.stochastic_block_model([size] * n_comm, P, seed=4)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
order = list(G.nodes())
comm = {v: v // size for v in order}
A = nx.to_numpy_array(G, nodelist=order)
W = A / A.sum(0)
deg = np.array([G.degree(v) for v in order])

seeds = [v for v in order if comm[v] == 0][:3]
p0 = np.zeros(len(order))
for s in seeds:
    p0[order.index(s)] = 1 / len(seeds)


def rwr(alpha, iters=300):
    p = p0.copy()
    for _ in range(iters):
        p = alpha * (W @ p) + (1 - alpha) * p0
    return p


# ── 그리기용 배치: 커뮤니티를 오각형으로 놓고, 시드 커뮤니티(0)를 왼쪽에 ─────────
pos = {}
for c in range(n_comm):
    ang = np.deg2rad(180 - 72 * c)
    cx, cy = np.cos(ang), np.sin(ang)
    sub = G.subgraph([v for v in order if comm[v] == c])
    sp = nx.spring_layout(sub, seed=4, k=0.35)
    for v, (x, y) in sp.items():
        pos[v] = (cx + 0.46 * x, cy + 0.46 * y)
intra = [(u, v) for u, v in G.edges() if comm[u] == comm[v]]
inter = [(u, v) for u, v in G.edges() if comm[u] != comm[v]]

SCORE = LinearSegmentedColormap.from_list(
    "score", [mix(PRIMARY, 0.13), STEPS[1], STEPS[0], PRIMARY, shade(PRIMARY, 0.3)])
VMIN, VMAX = 3e-4, 2e-2
norm = LogNorm(vmin=VMIN, vmax=VMAX, clip=True)

plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
fig = plt.figure(figsize=(8.4, 6.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1], hspace=0.30, wspace=0.04,
                      left=0.09, right=0.98, top=0.95, bottom=0.09)

show = [0.50, 0.85, 0.99]
marks = ["①", "②", "③"]
for i, alpha in enumerate(show):
    p = rwr(alpha)
    mass = sum(p[order.index(v)] for v in order if comm[v] == 0)
    ax = fig.add_subplot(gs[0, i])
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=inter, edge_color=LINE, width=0.35, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=intra, edge_color=LINE, width=0.55)
    srt = sorted(order, key=lambda v: p[order.index(v)])
    cols = [SCORE(norm(np.clip(p[order.index(v)], VMIN, VMAX))) for v in srt]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=srt, node_color=cols,
                           linewidths=0, node_size=20)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=seeds, node_color="none",
                           edgecolors=TEXT, linewidths=1.5, node_size=62)
    ax.set_aspect("equal"); ax.margins(0.03)
    ax.set_xticks([]); ax.set_yticks([])
    for spn in ax.spines.values(): spn.set_visible(False)
    ax.text(0.5, 1.02, f"{marks[i]} α = {alpha:.2f}", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=12.5, fontweight="bold", color=TEXT)
    ax.text(0.5, 0.0, f"시드 커뮤니티에 남은 확률 {mass:.2f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=11, color=TEXT)

# ── 아래: α 와 차수 상관 ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, :])
alphas = np.arange(0.1, 1.0, 0.02)
rho = [spearmanr(rwr(a, 200), deg).statistic for a in alphas]
ax.axvspan(0.5, 0.8, color=FAINT, lw=0, zorder=0)
ax.text(0.65, 0.05, "흔히 쓰는 값", ha="center", va="bottom", fontsize=11, color=TEXT2)
ax.plot(alphas, rho, color=PRIMARY, lw=2, zorder=3)

pts = [(a, spearmanr(rwr(a), deg).statistic) for a in show]
(a1, r1), (a2, r2), (a3, r3) = pts
ax.plot([alphas[-1], a3], [rho[-1], r3], color=PRIMARY, lw=1.4, ls=(0, (2, 2)), zorder=3)
ax.plot([a1, a2], [r1, r2], "o", ms=7, color=PRIMARY, zorder=4)
ax.plot([a3], [r3], "o", ms=8, color=ACCENT_INK, zorder=4)
ax.text(a1, r1 + 0.09, f"① ρ = {r1:.2f}", ha="center", va="bottom", fontsize=11,
        fontweight="bold", color=TEXT)
ax.text(a2 + 0.02, r2 - 0.06, f"② ρ = {r2:.2f}", ha="left", va="center", fontsize=11,
        fontweight="bold", color=TEXT)
ax.text(a3 - 0.025, r3, f"③ ρ = {r3:.2f} · 허브 줄세우기", ha="right", va="center",
        fontsize=11.5, fontweight="bold", color=ACCENT_INK)

ax.set_xlim(0.08, 1.02); ax.set_ylim(0, 1.05)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("restart α", fontsize=11.5)
ax.set_ylabel("차수와의 상관 ρ", fontsize=11.5)
ax.grid(axis="y", color=LINE, lw=0.6, zorder=0)

save(fig, img("c06", "rwr.png"))
for a in (0.3, 0.5, 0.7, 0.85, 0.95, 0.99):
    p = rwr(a)
    mass = sum(p[order.index(v)] for v in order if comm[v] == 0)
    print(f"  α={a:.2f}  시드커뮤니티 확률질량 {mass:.3f}  ρ(차수) {spearmanr(p, deg).statistic:.3f}")

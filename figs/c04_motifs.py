"""04 §3 — 3-노드 모티프와 그 유의성.

왼쪽: 방향 그래프에서 자주 이야기되는 3-노드 부분구조 네 가지(2×2).
오른쪽: 심어둔 feed-forward loop 가 방향 보존 재배선 null 대비 유의한가.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from itertools import combinations

# ── 모티프 도식 ───────────────────────────────────────────────────────
shapes = [
    ("Cascade", "연쇄", [("A", "B"), ("B", "C")]),
    ("Feed-forward loop", "FFL", [("A", "B"), ("B", "C"), ("A", "C")]),
    ("Feedback loop", "되먹임", [("A", "B"), ("B", "C"), ("C", "A")]),
    ("Fan-out", "공동 조절", [("A", "B"), ("A", "C")]),
]

# ── FFL 을 심은 합성 조절 네트워크 (old script verbatim) ───────────────
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

# ── 그림 ─────────────────────────────────────────────────────────────
W, Hh = 8.4, 4.25
fig = plt.figure(figsize=(W, Hh))

def inch_axes(x, y, w, h):
    return fig.add_axes([x / W, y / Hh, w / W, h / Hh])

NODE, EDGE = PRIMARY, "#9AA1AB"
R = 0.27
pos = {"A": (0.0, 0.85), "B": (-0.85, -0.5), "C": (0.85, -0.5)}
mw, mh = 1.75, 1.85
xr = 2.6
yr = xr * mh / mw
slots = [(0.1, 2.05), (1.95, 2.05), (0.1, 0.1), (1.95, 0.1)]
for (x0, y0), (name, sub, edges) in zip(slots, shapes):
    ax = inch_axes(x0, y0, mw, mh)
    ax.set_xlim(-xr / 2, xr / 2); ax.set_ylim(1.2 - yr, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.patch.set_alpha(0)
    for s, t in edges:
        p0, p1 = np.array(pos[s]), np.array(pos[t])
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        a0, a1 = p0 + d * (R + 0.03), p1 - d * (R + 0.04)
        ax.add_patch(mp.FancyArrowPatch(tuple(a0), tuple(a1), arrowstyle="-|>", mutation_scale=14,
                                        color=EDGE, lw=1.8, shrinkA=0, shrinkB=0, zorder=2))
    for k, (x, y) in pos.items():
        ax.add_patch(mp.Circle((x, y), R, fc=NODE, ec="none", zorder=3))
        ax.text(x, y, k, ha="center", va="center", fontsize=11.5, fontweight="bold",
                color=on(NODE), zorder=4)
    ax.text(0, -0.9, name, ha="center", va="top", fontsize=11.5, fontweight="bold", color=TEXT)
    ax.text(0, -1.23, sub, ha="center", va="top", fontsize=11, color=TEXT2)

fig.text(0.1 / W, 4.08 / Hh, "① 방향 그래프의 3-노드 패턴", ha="left", va="bottom",
         fontsize=12.5, fontweight="bold", color=TEXT)

# ── 히스토그램 ───────────────────────────────────────────────────────
ax = inch_axes(4.55, 0.62, 3.7, 3.25)
NULLC = STEPS[1]
counts, _, _ = ax.hist(null, bins=24, color=NULLC, edgecolor="white", linewidth=0.4, zorder=2)
hmax = counts.max()
ax.set_ylim(0, hmax * 1.3)
ax.set_xlim(null.min() - 2, obs + 3)
ax.axvline(null.mean(), color=PRIMARY, lw=1.4, ls=(0, (4, 3)), zorder=3)
ax.axvline(obs, color=ACCENT_INK, lw=2.4, zorder=3)
ax.text(null.mean() + 0.8, hmax * 1.17, f"null 평균 {null.mean():.1f}", ha="left", va="center",
        fontsize=11, color=PRIMARY, fontweight="bold")
ax.text(37, hmax * 0.5, f"null 분포\n재배선 {N}회", ha="left", va="center", fontsize=11,
        color=TEXT2, linespacing=1.35)
ax.text(obs - 1.2, hmax * 1.0, f"관측 {obs}\nZ = {z:.1f}", ha="right", va="center", fontsize=11.5,
        color=ACCENT_INK, fontweight="bold", linespacing=1.4)
ax.set_xlabel("FFL 개수", fontsize=11.5)
ax.set_ylabel("빈도", fontsize=11.5)
ax.tick_params(labelsize=11)
fig.text(4.25 / W, 4.08 / Hh, "② 심어둔 FFL은 유의하게 많은가", ha="left", va="bottom",
         fontsize=12.5, fontweight="bold", color=TEXT)

save(fig, img("c04", "motifs.png"))
print(f"노드 {D.number_of_nodes()}, 엣지 {D.number_of_edges()}")
print(f"관측 FFL {obs} | null 평균 {null.mean():.1f} ± {null.std(ddof=1):.1f} "
      f"| Z = {z:.2f} | p = {p:.4f}")

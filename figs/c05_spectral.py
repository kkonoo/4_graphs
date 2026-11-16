"""05 §4 — 스펙트럴 방법.

정규화 라플라시안의 고유값 스펙트럼과 피들러 벡터. 고유값 사이의 '틈'이
커뮤니티 수를 귀띔해 준다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from matplotlib.ticker import FuncFormatter

# 커뮤니티 4개짜리 그래프 (계산은 옛 스크립트 그대로)
n_comm, size, p_in, p_out = 4, 40, 0.18, 0.010
P = np.full((n_comm, n_comm), p_out); np.fill_diagonal(P, p_in)
G = nx.stochastic_block_model([size] * n_comm, P, seed=6)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
L = nx.normalized_laplacian_matrix(G).todense()
w, v = np.linalg.eigh(np.asarray(L))

NEUTRAL = "#9AA1AB"
NEG, POS = CAT[1], CAT[2]                 # 피들러 부호 두 쪽 (나란한 두 집단)
minus = FuncFormatter(lambda x, _: f"{x:g}".replace("-", "\u2212"))
plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})

fig = plt.figure(figsize=(8.4, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], hspace=0.62, wspace=0.14)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

def marker(ax, text, x=-0.2, y=1.07):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12.5, fontweight="bold", color=TEXT)

# --- ① 고유값 스펙트럼 ------------------------------------------------------
ax = ax1
k = 14
ax.plot(range(1, k + 1), w[:k], "-", color=NEUTRAL, lw=1.4, zorder=2)
ax.plot(range(n_comm + 1, k + 1), w[n_comm:k], "o", color=NEUTRAL, ms=5.5, zorder=3)
ax.plot(range(1, n_comm + 1), w[:n_comm], "o", color=PRIMARY, ms=8, zorder=4)
gap = w[n_comm] - w[n_comm - 1]
x_gap = n_comm + 1
ax.plot([n_comm, x_gap], [w[n_comm - 1]] * 2, color=ACCENT_INK, lw=1.1, ls=(0, (3, 2)), zorder=2)
ax.annotate("", xy=(x_gap, w[n_comm] - 0.012), xytext=(x_gap, w[n_comm - 1] + 0.004),
            arrowprops=dict(arrowstyle="<|-|>", color=ACCENT_INK, lw=1.4, mutation_scale=10,
                            shrinkA=0, shrinkB=0), zorder=5)
ax.text(x_gap + 0.45, (w[n_comm] + w[n_comm - 1]) / 2, f"틈 = {gap:.3f}", color=ACCENT_INK,
        fontsize=11.5, fontweight="bold", va="center", ha="left")
ax.text(1.9, 0.035, f"작은 고유값 {n_comm}개", color=PRIMARY, fontsize=11, fontweight="bold",
        va="center", ha="left")
ax.set_xticks([1, 4, 8, 12])
ax.set_yticks([0, 0.2, 0.4])
ax.set_ylim(-0.04, 0.58); ax.set_xlim(0.3, 14.7)
ax.yaxis.set_major_formatter(minus)
ax.set_xlabel("고유값 순번", fontsize=11.5)
ax.set_ylabel("고유값  λ", fontsize=11.5)
marker(ax, "① 정규화 라플라시안 스펙트럼")

# --- ② 피들러 벡터 ----------------------------------------------------------
ax = ax2
fiedler = np.asarray(v[:, 1]).ravel()
order = np.argsort(fiedler)
fs = fiedler[order]
xs = np.arange(len(fs))
ax.axhline(0, color=TEXT2, ls=(0, (4, 3)), lw=1.0, zorder=1)
ax.plot(xs[fs <= 0], fs[fs <= 0], "o", color=NEG, ms=3.2, mec="none", zorder=3)
ax.plot(xs[fs > 0], fs[fs > 0], "o", color=POS, ms=3.2, mec="none", zorder=3)
pos_n, neg_n = int((fiedler > 0).sum()), int((fiedler <= 0).sum())
ax.text(4, 0.065, f"음수 {neg_n}개", color=NEG, fontsize=11.5, fontweight="bold", ha="left", va="center")
ax.text(108, 0.135, f"양수 {pos_n}개", color=POS, fontsize=11.5, fontweight="bold", ha="right", va="center")
ax.set_yticks([-0.1, 0, 0.1])
ax.yaxis.set_major_formatter(minus)
ax.set_xticks([0, 50, 100, 150])
ax.set_xlabel("노드 (피들러 값 순으로 정렬)", fontsize=11.5)
ax.set_ylabel("피들러 값", fontsize=11.5)
marker(ax, "② 피들러 벡터")

# --- ③ 그래프 위에 표시 -----------------------------------------------------
ax = ax3
pos = nx.spring_layout(G, seed=6, k=0.28)
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LINE, width=0.5)
nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=[POS if f > 0 else NEG for f in fiedler],
                       linewidths=0, node_size=30)
ax.margins(0.05); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.text(0.0, 1.035, "③ 부호로 이등분한 결과", transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12.5, fontweight="bold", color=TEXT)
ax.text(0.48, -0.03, f"음수 {neg_n}", transform=ax.transAxes, ha="right", va="top",
        fontsize=11.5, fontweight="bold", color=NEG)
ax.text(0.5, -0.03, ":", transform=ax.transAxes, ha="center", va="top",
        fontsize=11.5, fontweight="bold", color=TEXT2)
ax.text(0.52, -0.03, f"양수 {pos_n}", transform=ax.transAxes, ha="left", va="top",
        fontsize=11.5, fontweight="bold", color=POS)

save(fig, img("c05", "spectral.png"))
print("가장 작은 고유값 8개:", np.round(w[:8], 4))
print(f"λ_{n_comm+1} − λ_{n_comm} = {gap:.4f}")
print(f"피들러 이등분: {neg_n} : {pos_n}")

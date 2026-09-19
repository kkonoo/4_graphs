"""06 §2 — 재시작 랜덤워크(RWR)와 restart 파라미터.

왼쪽 셋: 시드에서 출발한 확률이 네트워크로 퍼지는 모습.
오른쪽: alpha 를 1 에 가깝게 두면 결국 '차수 줄세우기'가 된다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx
from scipy.stats import spearmanr

OUT = outdir("c06")

n_comm, size, p_in, p_out = 5, 60, 0.12, 0.008
P = np.full((n_comm, n_comm), p_out); np.fill_diagonal(P, p_in)
G = nx.stochastic_block_model([size] * n_comm, P, seed=4)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
order = list(G.nodes())
comm = {v: v // size for v in order}
A = nx.to_numpy_array(G, nodelist=order)
W = A / A.sum(0)
deg = np.array([G.degree(v) for v in order])
pos = nx.spring_layout(G, seed=4, k=0.26)

seeds = [v for v in order if comm[v] == 0][:3]
p0 = np.zeros(len(order))
for s in seeds:
    p0[order.index(s)] = 1 / len(seeds)


def rwr(alpha, iters=300):
    p = p0.copy()
    for _ in range(iters):
        p = alpha * (W @ p) + (1 - alpha) * p0
    return p


fig = plt.figure(figsize=(15.6, 4.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.22)

for i, alpha in enumerate([0.50, 0.85, 0.99]):
    p = rwr(alpha)
    mass = sum(p[order.index(v)] for v in order if comm[v] == 0)
    ax = fig.add_subplot(gs[0, i])
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LGREY, width=0.4)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=p, cmap="YlOrRd",
                                   linewidths=0, node_size=40,
                                   vmin=0, vmax=p.max())
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=seeds, node_color="none",
                           edgecolors=NAVY, linewidths=2.0, node_size=110)
    ax.set_title(f"restart α = {alpha}", color=NAVY, fontsize=12)
    ax.text(0.5, -0.03,
            f"시드 커뮤니티에 남은 확률 {mass:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
            color=RED if mass < 0.4 else "#4A4A4A")
    ax.margins(0.05); ax.set_axis_off()

ax = fig.add_subplot(gs[0, 3])
alphas = np.arange(0.1, 1.0, 0.02)
rho = [spearmanr(rwr(a, 200), deg).statistic for a in alphas]
ax.plot(alphas, rho, color=NAVY, lw=2)
for a, col in [(0.50, TEAL), (0.85, ORANGE), (0.99, RED)]:
    ax.axvline(a, color=col, ls="--", lw=1.3)
ax.set_xlabel("restart α"); ax.set_ylabel("RWR 점수와 차수의 상관 (ρ)")
ax.set_ylim(0, 1)
ax.set_title("⚠ α가 1에 가까우면\n허브 줄세우기가 된다", color=NAVY, fontsize=12)

fig.suptitle("시드(테두리 표시) 3개에서 출발한 확률이 어디까지 퍼지는가",
             color=NAVY, fontsize=13, fontweight="bold", y=1.03)
fig.savefig(os.path.join(OUT, "rwr.png"))
print("wrote", os.path.join(OUT, "rwr.png"))
for a in (0.3, 0.5, 0.7, 0.85, 0.95, 0.99):
    p = rwr(a)
    mass = sum(p[order.index(v)] for v in order if comm[v] == 0)
    print(f"  α={a:.2f}  시드커뮤니티 확률질량 {mass:.3f}  ρ(차수) {spearmanr(p, deg).statistic:.3f}")

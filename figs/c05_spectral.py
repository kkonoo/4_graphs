"""05 §4 — 스펙트럴 방법.

라플라시안의 고유값 스펙트럼과 피들러 벡터. 고유값 사이의 '틈'이
커뮤니티 수를 귀띔해 준다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx

OUT = outdir("c05")

# 커뮤니티 4개짜리 그래프
n_comm, size, p_in, p_out = 4, 40, 0.18, 0.010
P = np.full((n_comm, n_comm), p_out); np.fill_diagonal(P, p_in)
G = nx.stochastic_block_model([size] * n_comm, P, seed=6)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
L = nx.normalized_laplacian_matrix(G).todense()
w, v = np.linalg.eigh(np.asarray(L))

fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), gridspec_kw={"wspace": 0.26})

# --- A. 고유값 스펙트럼 ----------------------------------------------------
ax = axes[0]
k = 14
ax.plot(range(1, k + 1), w[:k], "o-", color=NAVY, lw=1.6, ms=6)
ax.plot(range(1, n_comm + 1), w[:n_comm], "o", color=RED, ms=9)
gap = w[n_comm] - w[n_comm - 1]
ax.annotate(f"틈 = {gap:.3f}", xy=(n_comm + 0.5, (w[n_comm] + w[n_comm-1]) / 2),
            xytext=(n_comm + 2.4, (w[n_comm] + w[n_comm-1]) / 2 - 0.10),
            color=RED, fontsize=11,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
ax.set_xlabel("고유값 순번"); ax.set_ylabel("고유값  λ")
ax.set_title(f"정규화 라플라시안 스펙트럼 — 작은 고유값 {n_comm}개", color=NAVY,
             fontsize=12)

# --- B. 피들러 벡터 --------------------------------------------------------
ax = axes[1]
fiedler = np.asarray(v[:, 1]).ravel()
order = np.argsort(fiedler)
ax.plot(fiedler[order], ".", color=TEAL, ms=5)
ax.axhline(0, color=RED, ls="--", lw=1.4)
ax.set_xlabel("노드 (피들러 값 순으로 정렬)")
ax.set_ylabel("두 번째 고유벡터 값")
pos_n, neg_n = int((fiedler > 0).sum()), int((fiedler <= 0).sum())
ax.set_title(f"피들러 벡터 — 부호로 나누면 {neg_n} : {pos_n}", color=NAVY,
             fontsize=12)

# --- C. 그래프 위에 표시 ---------------------------------------------------
ax = axes[2]
pos = nx.spring_layout(G, seed=6, k=0.28)
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LGREY, width=0.5)
nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=[ORANGE if f > 0 else BLUE for f in fiedler],
                       linewidths=0, node_size=60)
ax.set_title("피들러 벡터의 부호로 이등분한 결과", color=NAVY, fontsize=12)
ax.margins(0.06); ax.set_axis_off()

fig.savefig(os.path.join(OUT, "spectral.png"))
print("wrote", os.path.join(OUT, "spectral.png"))
print("가장 작은 고유값 8개:", np.round(w[:8], 4))
print(f"λ_{n_comm+1} − λ_{n_comm} = {gap:.4f}")
print(f"피들러 이등분: {neg_n} : {pos_n}")

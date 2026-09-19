"""03 §1 — 차수 분포를 왜 log-log 로 보는가.

같은 노드 수·엣지 수인데 분포 모양이 다른 두 네트워크.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx

OUT = outdir("c03")

n, m_per = 3000, 2
BA = nx.barabasi_albert_graph(n, m_per, seed=1)
ER = nx.gnm_random_graph(n, BA.number_of_edges(), seed=1)

def degs(G):
    return np.array([d for _, d in G.degree()])

dBA, dER = degs(BA), degs(ER)

fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), gridspec_kw={"wspace": 0.28})

# --- A. 선형 축 ------------------------------------------------------------
ax = axes[0]
bins = np.arange(0, max(dBA.max(), dER.max()) + 2) - 0.5
ax.hist(dER, bins=bins, color=BLUE, alpha=0.75, label="랜덤 (ER)")
ax.hist(dBA, bins=bins, color=ORANGE, alpha=0.75, label="허브형 (BA)")
ax.set_xlim(0, 40)
ax.set_xlabel("차수  k"); ax.set_ylabel("노드 수")
ax.set_title("선형 축 — 꼬리가 안 보인다", color=NAVY, fontsize=12)
ax.legend(frameon=False)

# --- B. 같은 자료, log-log CCDF -------------------------------------------
ax = axes[1]
for d, col, lab in [(dER, BLUE, "랜덤 (ER)"), (dBA, ORANGE, "허브형 (BA)")]:
    x = np.sort(d)
    ccdf = 1.0 - np.arange(len(x)) / len(x)
    ax.loglog(x, ccdf, ".", color=col, ms=3.5, label=lab)
ax.set_xlabel("차수  k"); ax.set_ylabel("P(K ≥ k)")
ax.set_title("log-log — 꼬리가 드러난다", color=NAVY, fontsize=12)
ax.legend(frameon=False)

# --- C. 허브가 실제로 얼마나 큰가 ------------------------------------------
ax = axes[2]
top = 12
idx = np.argsort(dBA)[::-1][:top]
ax.bar(range(top), dBA[idx], color=ORANGE, label="허브형 (BA)")
ax.bar(range(top), np.sort(dER)[::-1][:top], color=BLUE, alpha=0.8,
       width=0.45, label="랜덤 (ER)")
ax.set_xticks(range(top)); ax.set_xticklabels([f"{i+1}" for i in range(top)],
                                              fontsize=9)
ax.set_xlabel("차수 상위 노드"); ax.set_ylabel("차수  k")
ax.set_title(f"최대 차수 {dBA.max()} vs {dER.max()}", color=NAVY, fontsize=12)
ax.legend(frameon=False)

fig.savefig(os.path.join(OUT, "degree_dist.png"))
print("wrote", os.path.join(OUT, "degree_dist.png"))
print(f"  노드 {n}, 엣지 {BA.number_of_edges()}")
print(f"  BA: 평균 {dBA.mean():.2f}, 최대 {dBA.max()}, 중앙 {np.median(dBA):.0f}")
print(f"  ER: 평균 {dER.mean():.2f}, 최대 {dER.max()}, 중앙 {np.median(dER):.0f}")
print(f"  평균 뭉침계수  BA {nx.average_clustering(BA):.4f} / ER {nx.average_clustering(ER):.4f}")

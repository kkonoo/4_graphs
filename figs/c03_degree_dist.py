"""03 §1 — 차수 분포를 왜 log-log 로 보는가.

같은 노드 수·엣지 수인데 분포 모양이 다른 두 네트워크 (모의실험).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx

n, m_per = 3000, 2
BA = nx.barabasi_albert_graph(n, m_per, seed=1)
ER = nx.gnm_random_graph(n, BA.number_of_edges(), seed=1)

def degs(G):
    return np.array([d for _, d in G.degree()])

dBA, dER = degs(BA), degs(ER)

BA_C, ER_C = PRIMARY, CAT[2]          # 허브형 · 랜덤 (나란한 두 네트워크, 세 패널에서 같은 색)

plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
fig = plt.figure(figsize=(8.4, 6.3))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.62, wspace=0.34)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, :])

def marker(ax, text):
    ax.text(0.0, 1.06, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12.5, fontweight="bold", color=TEXT)

# --- ① 선형 축 --------------------------------------------------------------
ax = ax1
bins = np.arange(0, max(dBA.max(), dER.max()) + 2) - 0.5
cER, _ = np.histogram(dER, bins=bins)
cBA, _ = np.histogram(dBA, bins=bins)
ks = np.arange(len(cER))
ax.bar(ks - 0.2, cER, width=0.4, color=ER_C, lw=0)
ax.bar(ks + 0.2, cBA, width=0.4, color=BA_C, lw=0)
ax.set_xlim(-0.5, 40)
ax.text(3.4, cBA[2], "허브형 (BA)", color=BA_C, fontsize=11.5, fontweight="bold", va="center")
ax.text(7.0, 560, "랜덤 (ER)", color=ER_C, fontsize=11.5, fontweight="bold", va="center")
ax.annotate("k > 20 쪽 꼬리는\n막대가 보이지 않음", xy=(28, 25), xytext=(26, 700),
            fontsize=11, color=TEXT2, ha="center", va="center", linespacing=1.35,
            arrowprops=dict(arrowstyle="-|>", color="#9AA1AB", lw=1.2))
ax.set_xlabel("차수  k", fontsize=11.5); ax.set_ylabel("노드 수", fontsize=11.5)
marker(ax, "① 선형 축")

# --- ② 같은 자료, log-log CCDF -----------------------------------------------
ax = ax2
for d, col in [(dER, ER_C), (dBA, BA_C)]:
    x = np.sort(d)
    ccdf = 1.0 - np.arange(len(x)) / len(x)
    ax.loglog(x, ccdf, ".", color=col, ms=3.5)
ax.text(22, 0.06, "허브형 (BA)", color=BA_C, fontsize=11.5, fontweight="bold", ha="left", va="center")
ax.text(9.5, 6e-4, "랜덤 (ER)", color=ER_C, fontsize=11.5, fontweight="bold", ha="right", va="center")
ax.set_xlabel("차수  k", fontsize=11.5); ax.set_ylabel("P(K ≥ k)", fontsize=11.5)
marker(ax, "② log-log CCDF")

# --- ③ 차수 상위 노드 ---------------------------------------------------------
ax = ax3
top = 12
idx = np.argsort(dBA)[::-1][:top]
topBA = dBA[idx]
topER = np.sort(dER)[::-1][:top]
xs = np.arange(top)
bw = 0.38
ax.bar(xs - bw / 2, topBA, color=BA_C, width=bw, lw=0)
ax.bar(xs + bw / 2, topER, color=ER_C, width=bw, lw=0)
ax.text(-bw / 2, topBA[0] + 3, f"{dBA.max()}", ha="center", va="bottom", fontsize=11.5,
        fontweight="bold", color=BA_C)
ax.text(bw / 2, topER[0] + 3, f"{dER.max()}", ha="center", va="bottom", fontsize=11.5,
        fontweight="bold", color=ER_C)
ax.text(7.5, 125, "허브형 (BA)", color=BA_C, fontsize=11.5, fontweight="bold", ha="left", va="center")
ax.text(7.5, 105, "랜덤 (ER)", color=ER_C, fontsize=11.5, fontweight="bold", ha="left", va="center")
ax.add_patch(mp.Rectangle((7.0, 120), 0.3, 10, color=BA_C, lw=0))
ax.add_patch(mp.Rectangle((7.0, 100), 0.3, 10, color=ER_C, lw=0))
ax.set_xticks(xs); ax.set_xticklabels([f"{i+1}" for i in range(top)])
ax.set_xlim(-0.7, top - 0.3)
ax.set_ylim(0, 160)
ax.set_xlabel("차수 상위 노드 (순위)", fontsize=11.5); ax.set_ylabel("차수  k", fontsize=11.5)
marker(ax, "③ 차수 상위 12개 노드")

save(fig, img("c03", "degree_dist.png"))
print(f"  노드 {n}, 엣지 {BA.number_of_edges()}")
print(f"  BA: 평균 {dBA.mean():.2f}, 최대 {dBA.max()}, 중앙 {np.median(dBA):.0f}")
print(f"  ER: 평균 {dER.mean():.2f}, 최대 {dER.max()}, 중앙 {np.median(dER):.0f}")
print(f"  평균 뭉침계수  BA {nx.average_clustering(BA):.4f} / ER {nx.average_clustering(ER):.4f}")

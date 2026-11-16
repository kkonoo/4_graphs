"""06 §3 — 링크 예측 평가의 함정.

엣지 10%를 숨기고 heuristic 네 가지로 복원. 음성 샘플을 어떻게 잡느냐에 따라
AUROC 는 그대로인데 AUPRC 는 무너진다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)

rng = np.random.default_rng(0)

G = nx.powerlaw_cluster_graph(800, 3, 0.35, seed=1)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
edges = list(G.edges()); m = len(edges)
deg = dict(G.degree())

n_test = int(0.1 * m)
test_pos = [edges[i] for i in rng.choice(m, n_test, replace=False)]
H = G.copy(); H.remove_edges_from(test_pos)
nodes = list(H.nodes())


def sample_neg(k):
    out = set()
    while len(out) < k:
        u, v = rng.choice(len(nodes), 2, replace=False)
        u, v = nodes[u], nodes[v]
        if u != v and not G.has_edge(u, v):
            out.add((u, v))
    return list(out)


scorers = {
    "Common Neighbors": lambda p: [len(list(nx.common_neighbors(H, u, v))) for u, v in p],
    "Jaccard": lambda p: [s for _, _, s in nx.jaccard_coefficient(H, p)],
    "Adamic–Adar": lambda p: [s for _, _, s in nx.adamic_adar_index(H, p)],
    "Preferential Att.": lambda p: [s for _, _, s in nx.preferential_attachment(H, p)],
}
names = list(scorers)
cols = {n: CAT[i] for i, n in enumerate(names)}      # 나란한 네 방법
NEUTRAL = "#9AA1AB"

settings = [("음성을 양성과 같은 수만 뽑으면", n_test),
            ("음성을 현실 비율대로 뽑으면 (1:50)", n_test * 50)]
markers = ["① 음성을 양성과 같은 수만 (1:1)", "② 음성을 현실 비율대로 (1:50)"]

plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
fig = plt.figure(figsize=(8.4, 6.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.4, wspace=0.26)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
summary = {}

for si, (lab, n_neg) in enumerate(settings):
    neg = sample_neg(n_neg)
    pairs = test_pos + neg
    y = np.r_[np.ones(len(test_pos)), np.zeros(len(neg))]
    summary[lab] = {}
    ax = axes[si]
    for name, f in scorers.items():
        s = np.array(f(pairs), dtype=float)
        pr, rc, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        auc = roc_auc_score(y, s)
        summary[lab][name] = (auc, ap)
        ax.plot(rc, pr, color=cols[name], lw=1.8, label=f"{name}  {ap:.2f}")
    chance_lab = f"우연 수준  {y.mean():.3f}"
    ax.axhline(y.mean(), color=NEUTRAL, ls="--", lw=1.3,
               label=(chance_lab if si == 1 else "_nolegend_"))
    if si == 0:                                   # 범례와 겹치지 않게 점선 바로 위 왼쪽 빈 곳에 직접 라벨
        ax.text(0.0, y.mean() + 0.025, chance_lab, fontsize=11, color=TEXT2,
                ha="left", va="bottom")
    ax.set_xlabel("Recall", fontsize=11.5); ax.set_ylabel("Precision", fontsize=11.5)
    ax.set_ylim(0, 1.02); ax.set_xlim(-0.02, 1.02)
    ax.text(0.5, 1.04, markers[si], transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT)
    if si == 0:
        lk = dict(loc="lower right", bbox_to_anchor=(1.0, 0.0))
    else:
        lk = dict(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.legend(frameon=False, fontsize=11, title="AUPRC", title_fontsize=11,
              handlelength=1.3, labelspacing=0.25, borderaxespad=0.15, borderpad=0.2,
              alignment="left", **lk)

# --- ③ 요약: AUROC 는 그대로, AUPRC 만 무너진다 ---------------------------
ax = axes[2]
x = np.arange(len(names)); w = 0.19
bal, real = settings[0][0], settings[1][0]
auc_b = [summary[bal][n][0] for n in names]
auc_r = [summary[real][n][0] for n in names]
ap_b = [summary[bal][n][1] for n in names]
ap_r = [summary[real][n][1] for n in names]
ax.bar(x - 1.5*w, auc_b, w, color="#5F6672", label="AUROC (균형)")   # 무채색: 위 네 방법의 색과 겹치지 않게
ax.bar(x - 0.5*w, auc_r, w, color="#8B929D", label="AUROC (현실)")
ax.bar(x + 0.5*w, ap_b, w, color="#BCC2CB", label="AUPRC (균형)")
ax.bar(x + 1.5*w, ap_r, w, color=ACCENT, label="AUPRC (현실)")   # 놓치면 안 되는 하나
for xi, v in zip(x, ap_r):
    ax.text(xi + 1.5*w, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=11,
            fontweight="bold", color=ACCENT_INK)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("점수", fontsize=11.5)
ax.yaxis.grid(True, color=LINE, lw=0.8); ax.set_axisbelow(True)
ax.text(0.5, 1.04, "③ 네 방법 요약", transform=ax.transAxes, ha="center", va="bottom",
        fontsize=12, fontweight="bold", color=TEXT)
ax.legend(frameon=False, fontsize=11, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, 1.01), handlelength=1.2, columnspacing=1.4)

save(fig, img("c06", "linkpred.png"))
print(f"노드 {G.number_of_nodes()}  엣지 {G.number_of_edges()}  밀도 {nx.density(G):.4f}")
for lab in summary:
    for n in names:
        a, p = summary[lab][n]
        print(f"  {lab:<34} {n:<18} AUROC {a:.3f}  AUPRC {p:.3f}")

# --- 차수 편향 -------------------------------------------------------------
cand = sample_neg(20000)
aa = sorted(nx.adamic_adar_index(H, cand), key=lambda t: -t[2])
top_deg = np.mean([(deg[u] + deg[v]) / 2 for u, v, _ in aa[:200]])
bot_deg = np.mean([(deg[u] + deg[v]) / 2 for u, v, _ in aa[-200:]])
print(f"\n전체 평균 차수 {np.mean(list(deg.values())):.2f} | "
      f"AA 상위 200쌍 {top_deg:.2f} | AA 하위 200쌍 {bot_deg:.2f}")

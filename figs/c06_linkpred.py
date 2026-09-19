"""06 §3 — 링크 예측 평가의 함정.

엣지 10%를 숨기고 heuristic 네 가지로 복원. 음성 샘플을 어떻게 잡느냐에 따라
AUROC 는 그대로인데 AUPRC 는 무너진다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)

OUT = outdir("c06")
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
cols = {"Common Neighbors": BLUE, "Jaccard": PURPLE,
        "Adamic–Adar": TEAL, "Preferential Att.": ORANGE}

settings = [("음성을 양성과 같은 수만 뽑으면", n_test),
            ("음성을 현실 비율대로 뽑으면 (1:50)", n_test * 50)]

fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7), gridspec_kw={"wspace": 0.28})
summary = {}

for si, (label, n_neg) in enumerate(settings):
    neg = sample_neg(n_neg)
    pairs = test_pos + neg
    y = np.r_[np.ones(len(test_pos)), np.zeros(len(neg))]
    summary[label] = {}
    ax = axes[si]
    for name, f in scorers.items():
        s = np.array(f(pairs), dtype=float)
        pr, rc, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        auc = roc_auc_score(y, s)
        summary[label][name] = (auc, ap)
        ax.plot(rc, pr, color=cols[name], lw=1.8,
                label=f"{name}  AUPRC={ap:.2f}")
    ax.axhline(y.mean(), color=GREY, ls="--", lw=1.3,
               label=f"우연 수준 {y.mean():.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title(label, color=NAVY, fontsize=12)
    ax.legend(frameon=False, fontsize=9)

# --- 세 번째 패널: AUROC 는 그대로인데 AUPRC 만 무너진다 -------------------
ax = axes[2]
names = list(scorers)
x = np.arange(len(names)); w = 0.2
bal, real = settings[0][0], settings[1][0]
ax.bar(x - 1.5*w, [summary[bal][n][0] for n in names], w, color=NAVY,
       label="AUROC (균형)")
ax.bar(x - 0.5*w, [summary[real][n][0] for n in names], w, color="#8FA3B0",
       label="AUROC (현실)")
ax.bar(x + 0.5*w, [summary[bal][n][1] for n in names], w, color=TEAL,
       label="AUPRC (균형)")
ax.bar(x + 1.5*w, [summary[real][n][1] for n in names], w, color=RED,
       label="AUPRC (현실)")
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
ax.set_ylim(0, 1)
ax.set_title("⚠ AUROC는 꿈쩍 않고 AUPRC만 무너진다", color=NAVY, fontsize=12)
ax.legend(frameon=False, fontsize=9, ncol=2)

fig.savefig(os.path.join(OUT, "linkpred.png"))
print("wrote", os.path.join(OUT, "linkpred.png"))
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

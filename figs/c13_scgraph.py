"""13 §3 — 논문 읽는 눈 실측: PBMC 3k 세포 kNN 그래프 위의 GCN.

특징: PCA 50차원 (pbmc3k_processed의 obsm X_pca, scanpy 튜토리얼 값)
그래프: PCA 50차원에서 세포마다 최근접 k개를 잇고 대칭화 (k = 5, 15, 50)
라벨 A: 세포유형 8개 (obs louvain — 튜토리얼이 kNN 그래프 위 군집에 이름을 붙인 것)
라벨 B: 그래프로 만든 라벨 — k = 15 그래프 위 Leiden(resolution 1.0) 군집, 생물학적 이름 없음 (k = 15에서만 평가)
학습 라벨 비율: 세포의 1% / 5% / 20% (유형마다 최소 1개), validation 10%, 나머지 test. 시드 5개
모델 (모두 validation으로 고름)
  - 로지스틱 회귀: PCA 50만 (C)
  - 레이블 전파: 그래프만 (06의 RWR, alpha)
  - GCN: PCA 50 + 그래프, 2층 은닉 64, dropout 0.5, Adam lr 0.01, 200 epoch 중 validation 최고 시점
지표: balanced accuracy. 결과: figs/cache/c13_sc.csv
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree, homophily
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from c10_data import load_pbmc, CACHE
from c12_gnn import norm_adj

torch.set_num_threads(1)
OUT = outdir("c13")


def knn_edges(Z, k):
    A = kneighbors_graph(Z, k, include_self=False)
    A = ((A + A.T) > 0).tocoo()
    return torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)


def split(y, frac, seed, n_val=0.1):
    rng = np.random.default_rng(seed)
    tr = np.concatenate([rng.permutation(np.where(y == c)[0])[:max(1, round(frac * (y == c).sum()))]
                         for c in np.unique(y)])
    rest = rng.permutation(np.setdiff1d(np.arange(len(y)), tr))
    nv = int(n_val * len(y))
    return tr, rest[:nv], rest[nv:]


class GCN(nn.Module):
    def __init__(self, i, o, h=64):
        super().__init__(); self.c1, self.c2 = GCNConv(i, h), GCNConv(h, o)

    def forward(self, x, ei):
        x = F.dropout(F.relu(self.c1(F.dropout(x, 0.5, self.training), ei)), 0.5, self.training)
        return self.c2(x, ei)


def fit_gcn(X, y, ei, tr, va, te, seed, epochs=200):
    torch.manual_seed(seed)
    m = GCN(X.shape[1], int(y.max()) + 1)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    best = (-1, 0)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(Xt, ei)[tr], yt[tr]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = m(Xt, ei).argmax(1).numpy()
        v = bacc(y[va], p[va])
        if v > best[0]:
            best = (v, bacc(y[te], p[te]))
    return best[1]


def fit_lr(X, y, tr, va, te):
    best = max((LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr]) for C in (0.1, 1, 10)),
               key=lambda m: bacc(y[va], m.predict(X[va])))
    return bacc(y[te], best.predict(X[te]))


def fit_lp(A, y, tr, va, te):
    Y0 = torch.zeros(len(y), int(y.max()) + 1); Y0[tr, y[tr]] = 1
    best = (-1, 0)
    for a in (0.5, 0.7, 0.9, 0.99):
        Y = Y0.clone()
        for _ in range(50):
            Y = a * torch.sparse.mm(A, Y) + (1 - a) * Y0
        p = Y.argmax(1).numpy()
        v = bacc(y[va], p[va])
        if v > best[0]:
            best = (v, bacc(y[te], p[te]))
    return best[1]


def leiden_labels(ei, n, res=1.0):
    import igraph as ig, leidenalg as la
    e = ei.t().numpy(); e = e[e[:, 0] < e[:, 1]]
    g = ig.Graph(n=n, edges=e.tolist())
    return np.array(la.find_partition(g, la.RBConfigurationVertexPartition,
                                      resolution_parameter=res, seed=0).membership)


if __name__ == "__main__":
    f = os.path.join(CACHE, "c13_sc.csv")
    _, lab, Z = load_pbmc()
    Z = Z.astype(np.float32)
    types = np.unique(lab); yA = np.searchsorted(types, lab)
    E = {k: knn_edges(Z, k) for k in (5, 15, 50)}
    yB = leiden_labels(E[15], len(Z))
    print("세포유형", len(types), "그래프 라벨 군집 수", yB.max() + 1)
    for k, ei in E.items():
        print(f"k={k}: 엣지 {ei.shape[1] // 2}, 세포유형 homophily {homophily(ei, torch.tensor(yA)):.3f}, "
              f"그래프 라벨 homophily {homophily(ei, torch.tensor(yB)):.3f}")
    GRID = [("세포유형", yA, k, fr) for k in (5, 15, 50) for fr in (0.01, 0.05, 0.2)] + \
           [("그래프 라벨", yB, 15, fr) for fr in (0.01, 0.05, 0.2)]
    done = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=["labels", "k", "frac"])
    rows = done.to_dict("records")
    todo = [g for g in GRID if len(done[(done.labels == g[0]) & (done.k == g[2]) & (done.frac == g[3])]) < 15]
    for labname, y, k, frac in todo:                      # 이미 저장된 조합은 건너뜀
        ei = E[k]; A = norm_adj(ei, len(Z))
        t = time.time()
        for s in range(5):
            tr, va, te = split(y, frac, s)
            for name, fn in [("로지스틱 회귀 (PCA만)", lambda: fit_lr(Z, y, tr, va, te)),
                             ("레이블 전파 (그래프만)", lambda: fit_lp(A, y, tr, va, te)),
                             ("GCN (PCA + 그래프)", lambda: fit_gcn(Z, y, ei, tr, va, te, s))]:
                rows.append(dict(labels=labname, k=k, frac=frac, seed=s, model=name,
                                 n_train=len(tr), bacc=fn()))
        d = pd.DataFrame(rows[-15:]).groupby("model").bacc.mean().round(3).to_dict()
        print(labname, k, frac, d, f"{time.time() - t:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f, index=False)

    # ── 그림 ──────────────────────────────────────────────────────────
    d = pd.read_csv(f)
    M = [("로지스틱 회귀 (PCA만)", GREY, "s"), ("레이블 전파 (그래프만)", OLIVE, "D"),
         ("GCN (PCA + 그래프)", TEAL, "o")]
    g = d.groupby(["labels", "k", "frac", "model"]).bacc.agg(["mean", "std"]).reset_index()
    pd.set_option("display.width", 200)
    print(g.pivot_table(index=["labels", "k", "frac"], columns="model", values="mean").round(3))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.7), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    ax = axes[0]
    for m, c, mk in M:
        s_ = g[(g.labels == "세포유형") & (g.k == 15) & (g.model == m)]
        ax.errorbar(s_.frac * 100, s_["mean"], yerr=s_["std"], color=c, marker=mk, lw=2.3, capsize=3, label=m)
    ax.set_xscale("log"); ax.set_xticks([1, 5, 20]); ax.set_xticklabels(["1%", "5%", "20%"])
    ax.set_xlabel("라벨을 아는 세포의 비율"); ax.set_ylabel("balanced accuracy (test)")
    ax.set_ylim(0.6, 1.0); ax.set_title("세포유형, k = 15", color=NAVY)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax = axes[1]
    for m, c, mk in M:
        s_ = g[(g.labels == "세포유형") & (g.frac == 0.05) & (g.model == m)]
        ax.errorbar(np.log(s_.k), s_["mean"], yerr=s_["std"], color=c, marker=mk, lw=2.3, capsize=3, label=m)
    ax.set_xticks(np.log([5, 15, 50])); ax.set_xticklabels(["5", "15", "50"])
    ax.set_xlabel("kNN 그래프의 k"); ax.set_ylim(0.6, 1.0)
    ax.set_title("같은 세포, k만 바꾸면 (라벨 5%)", color=NAVY)
    ax = axes[2]
    w = 0.26
    for j, (m, c, _) in enumerate(M):
        v = [g[(g.labels == L) & (g.k == 15) & (g.frac == 0.05) & (g.model == m)]["mean"].iloc[0]
             for L in ("세포유형", "그래프 라벨")]
        xs = np.arange(2) + (j - 1) * w
        ax.bar(xs, v, w, color=c, label=m)
        for x_, y_ in zip(xs, v):
            ax.text(x_, y_ + 0.012, f"{y_:.2f}", ha="center", fontsize=9.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["세포유형\n(마커 유전자로 이름 붙인 군집)", "그래프 라벨\n(같은 그래프의 Leiden 군집)"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("balanced accuracy (test)")
    ax.set_title("라벨이 그래프에서 나왔다면 (k = 15, 라벨 5%)", color=NAVY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scgraph.png"))
    print("done")

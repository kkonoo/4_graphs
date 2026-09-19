"""12 — 그래프 신경망 실측 (Cora, PyG 2.x, CPU).

Cora: 논문 2,708개, 인용 엣지 5,278개, 단어 특징 1,433개, 분야 7개.
public split (학습 140 / 검증 500 / test 1,000)과 무작위 split(분야마다 학습 20개) 10회.
특징은 행 정규화(T.NormalizeFeatures). 신경망은 200 epoch 전체 배치 Adam,
검증 정확도가 가장 높은 epoch의 test 정확도를 보고.

모델
  - 로지스틱 회귀: 노드 특징만 (C는 검증으로)
  - MLP: 1433 -> 64 -> 7, dropout 0.5, lr 0.01, weight decay 5e-4
  - 레이블 전파: 특징 없이 그래프만. Y <- a*Â*Y + (1-a)*Y0 (06의 RWR, a는 검증으로)
  - 특징 전파 + 로지스틱 (SGC): Â^2 X 로 로지스틱 회귀 — 학습되는 W가 하나뿐인 GNN
  - GCN: 1433 -> 16 -> 7 (Kipf & Welling 2017 설정)
  - GraphSAGE (mean): 1433 -> 16 -> 7
  - GAT: 8 head x 8 -> 7, dropout 0.6, lr 0.005

실험 1 (c12_cora.csv): 위 모델들, public split 시드 10개 + 무작위 split 10개
실험 2 (c12_smooth_prop.csv): 학습 없이 Â^K X (K = 0..64) — 노드 표현 간 평균 코사인 거리, 로지스틱 정확도
실험 3 (c12_smooth_depth.csv): GCN 층 수 1-16 (은닉 64), 시드 5
실험 4 (c12_hetero.csv): 엣지 일부를 다른 분야끼리 잇는 무작위 엣지로 바꿔 homophily를 낮춤 — GCN vs SAGE vs MLP, 시드 5
"""
import sys, os, time, copy, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_undirected, homophily, add_self_loops, degree
from sklearn.linear_model import LogisticRegression

torch.set_num_threads(2)
OUT = outdir("c12")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DATA = os.path.join(CACHE, "Planetoid")


def load():
    return Planetoid(DATA, "Cora", transform=T.NormalizeFeatures())[0]


def random_split(data, seed, per_class=20, n_val=500, n_test=1000):
    g = torch.Generator().manual_seed(seed)
    y = data.y; n = len(y)
    tr = torch.cat([torch.nonzero(y == c).ravel()[torch.randperm(int((y == c).sum()), generator=g)[:per_class]]
                    for c in range(int(y.max()) + 1)])
    rest = torch.tensor([i for i in torch.randperm(n, generator=g).tolist() if i not in set(tr.tolist())])
    masks = []
    for idx in (tr, rest[:n_val], rest[n_val:n_val + n_test]):
        m = torch.zeros(n, dtype=torch.bool); m[idx] = True; masks.append(m)
    return masks


class MLP(nn.Module):
    def __init__(self, i, o, h=64, p=0.5):
        super().__init__(); self.l1, self.l2, self.p = nn.Linear(i, h), nn.Linear(h, o), p

    def forward(self, x, ei):
        x = F.dropout(x, self.p, self.training)
        return self.l2(F.dropout(F.relu(self.l1(x)), self.p, self.training))


class GNN(nn.Module):
    """conv 층을 layers개 쌓은 GCN/SAGE. 마지막 층 직전 표현을 self.h에 남김"""
    def __init__(self, i, o, h=16, layers=2, kind="gcn", p=0.5):
        super().__init__()
        Conv = {"gcn": GCNConv, "sage": SAGEConv}[kind]
        dims = [i] + [h] * (layers - 1) + [o]
        self.convs = nn.ModuleList([Conv(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.p = p

    def forward(self, x, ei):
        for k, conv in enumerate(self.convs):
            x = F.dropout(x, self.p, self.training)
            if k == len(self.convs) - 1:
                self.h = x
            x = conv(x, ei)
            if k < len(self.convs) - 1:
                x = F.relu(x)
        return x


class GAT(nn.Module):
    def __init__(self, i, o, h=8, heads=8, p=0.6):
        super().__init__()
        self.c1 = GATConv(i, h, heads=heads, dropout=p)
        self.c2 = GATConv(h * heads, o, heads=1, concat=False, dropout=p)
        self.p = p

    def forward(self, x, ei):
        x = F.dropout(x, self.p, self.training)
        x = F.elu(self.c1(x, ei))
        return self.c2(F.dropout(x, self.p, self.training), ei)


def train_nn(model, data, masks, lr=0.01, wd=5e-4, epochs=200, ei=None):
    ei = data.edge_index if ei is None else ei
    tr, va, te = masks
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = (-1, 0, None)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(data.x, ei)[tr], data.y[tr]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(data.x, ei).argmax(1)
        v = (pred[va] == data.y[va]).float().mean().item()
        if v > best[0]:
            best = (v, (pred[te] == data.y[te]).float().mean().item(), ep)
    return best      # (검증 정확도, 그때의 test 정확도, epoch)


def norm_adj(ei, n):
    ei, _ = add_self_loops(ei, num_nodes=n)
    d = degree(ei[0], n)
    w = d[ei[0]].rsqrt() * d[ei[1]].rsqrt()
    return torch.sparse_coo_tensor(ei, w, (n, n)).coalesce()


def fit_lr(X, y, masks):
    tr, va, te = [m.numpy() for m in masks]
    best = max((LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr]) for C in (0.1, 1, 10, 100)),
               key=lambda m: m.score(X[va], y[va]))
    return best.score(X[va], y[va]), best.score(X[te], y[te])


def label_prop(A, y, masks, n_cls=7):
    tr, va, te = masks
    Y0 = torch.zeros(len(y), n_cls); Y0[tr, y[tr]] = 1
    best = (-1, 0)
    for a in (0.5, 0.7, 0.9, 0.99):
        Y = Y0.clone()
        for _ in range(50):
            Y = a * torch.sparse.mm(A, Y) + (1 - a) * Y0
        pred = Y.argmax(1)
        v = (pred[va] == y[va]).float().mean().item()
        if v > best[0]:
            best = (v, (pred[te] == y[te]).float().mean().item())
    return best


def mean_cos_dist(H):
    """모든 노드 쌍의 평균 코사인 거리 (1 - cos). 0이면 모든 노드가 같은 방향"""
    Hn = H / H.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = len(H); s = Hn.sum(0)
    return float(1 - (s @ s - n) / (n * (n - 1)))


def run_models(data, masks, seed, A):
    y = data.y
    X = data.x.numpy()
    out = {}
    out["로지스틱 (특징만)"] = fit_lr(X, y.numpy(), masks)[1]
    torch.manual_seed(seed); out["MLP (특징만)"] = train_nn(MLP(1433, 7), data, masks)[1]
    out["레이블 전파 (그래프만)"] = label_prop(A, y, masks)[1]
    AX = torch.sparse.mm(A, torch.sparse.mm(A, data.x)).numpy()
    out["특징 전파 + 로지스틱"] = fit_lr(AX, y.numpy(), masks)[1]
    torch.manual_seed(seed); out["GCN"] = train_nn(GNN(1433, 7, kind="gcn"), data, masks)[1]
    torch.manual_seed(seed); out["GraphSAGE"] = train_nn(GNN(1433, 7, kind="sage"), data, masks)[1]
    torch.manual_seed(seed); out["GAT"] = train_nn(GAT(1433, 7), data, masks, lr=0.005)[1]
    return out


if __name__ == "__main__":
    data = load(); n = data.num_nodes
    A = norm_adj(data.edge_index, n)
    print(data, "homophily", round(homophily(data.edge_index, data.y), 3), flush=True)
    pub = (data.train_mask, data.val_mask, data.test_mask)
    f1 = os.path.join(CACHE, "c12_cora.csv")
    if not os.path.exists(f1):
        rows = []
        for s in range(10):
            t = time.time()
            for split, masks in [("public", pub), ("random", random_split(data, s))]:
                for k, v in run_models(data, masks, s, A).items():
                    rows.append(dict(split=split, seed=s, model=k, acc=v))
            print(s, f"{time.time()-t:.0f}s", {r["model"]: round(r["acc"], 3) for r in rows[-7:]}, flush=True)
            pd.DataFrame(rows).to_csv(f1, index=False)
    f2 = os.path.join(CACHE, "c12_smooth_prop.csv")
    if not os.path.exists(f2):
        rows, H = [], data.x.clone()
        for K in range(65):
            if K in (0, 1, 2, 4, 8, 16, 32, 64):
                rows.append(dict(K=K, dist=mean_cos_dist(H), acc=fit_lr(H.numpy(), data.y.numpy(), pub)[1]))
                print(rows[-1], flush=True)
            H = torch.sparse.mm(A, H)
        pd.DataFrame(rows).to_csv(f2, index=False)
    f3 = os.path.join(CACHE, "c12_smooth_depth.csv")
    if not os.path.exists(f3):
        rows = []
        for L in (1, 2, 4, 8, 16):
            for s in range(5):
                torch.manual_seed(s)
                m = GNN(1433, 7, h=64, layers=L, kind="gcn")
                va, te, ep = train_nn(m, data, pub)
                m.eval()
                with torch.no_grad():
                    m(data.x, data.edge_index)
                rows.append(dict(layers=L, seed=s, acc=te, dist=mean_cos_dist(m.h) if L > 1 else np.nan))
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(f3, index=False)
    f4 = os.path.join(CACHE, "c12_hetero.csv")
    if not os.path.exists(f4):
        rows = []
        und = data.edge_index[:, data.edge_index[0] < data.edge_index[1]]
        y = data.y
        for frac in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
            for s in range(5):
                g = torch.Generator().manual_seed(s)
                keep = torch.rand(und.shape[1], generator=g) >= frac
                k_new = int((~keep).sum())
                exist = set(map(tuple, und[:, keep].t().tolist()))
                new = []
                while len(new) < k_new:
                    u, v = torch.randint(n, (2,), generator=g).tolist()
                    e = (min(u, v), max(u, v))
                    if u != v and y[u] != y[v] and e not in exist:
                        exist.add(e); new.append(e)
                E = torch.cat([und[:, keep], torch.tensor(new, dtype=torch.long).t().reshape(2, -1)], 1)
                E = to_undirected(E)
                h = homophily(E, y)
                for name, mk, lr in [("GCN", lambda: GNN(1433, 7, kind="gcn"), 0.01),
                                     ("GraphSAGE", lambda: GNN(1433, 7, kind="sage"), 0.01),
                                     ("MLP", lambda: MLP(1433, 7), 0.01)]:
                    torch.manual_seed(s)
                    rows.append(dict(frac=frac, seed=s, homophily=h, model=name,
                                     acc=train_nn(mk(), data, pub, lr=lr, ei=E)[1]))
                print(frac, s, round(h, 3), [round(r["acc"], 3) for r in rows[-3:]], flush=True)
        pd.DataFrame(rows).to_csv(f4, index=False)

    # ── 그림 1: Cora 모델 비교 ───────────────────────────────────────
    d = pd.read_csv(f1)
    order = ["로지스틱 (특징만)", "MLP (특징만)", "레이블 전파 (그래프만)", "특징 전파 + 로지스틱",
             "GCN", "GraphSAGE", "GAT"]
    col = {"로지스틱 (특징만)": GREY, "MLP (특징만)": GREY, "레이블 전파 (그래프만)": OLIVE,
           "특징 전파 + 로지스틱": ORANGE, "GCN": TEAL, "GraphSAGE": BLUE, "GAT": PURPLE}
    g = d.groupby(["split", "model"]).acc.agg(["mean", "std", "min", "max"]).round(3)
    print(g)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    fig.suptitle("점 하나 = 시드(또는 split) 하나, 세로 막대 = 평균", fontsize=10.5, color="#555", y=0.02)
    for ax, (sp, title) in zip(axes, [("public", "public split (학습 140개 고정), 시드 10개"),
                                      ("random", "무작위 split 10개 (분야마다 학습 20개)")]):
        for i, m in enumerate(order):
            v = d[(d.split == sp) & (d.model == m)].acc.values
            ax.scatter(v, np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), s=22, color=col[m],
                       alpha=0.8, zorder=2)
            ax.plot([v.mean()] * 2, [i - 0.32, i + 0.32], color=NAVY, lw=3, zorder=3)
            ax.text(v.max() + 0.008, i, f"{v.mean():.3f}", va="center", fontsize=10)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
        ax.invert_yaxis(); ax.set_xlim(0.5, 0.9); ax.grid(axis="x", color=LGREY, lw=0.8)
        ax.set_xlabel("test 정확도"); ax.set_title(title, color=NAVY, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cora.png"))

    # ── 그림 2: 한계 — 과평활화, 층 수, homophily ────────────────────
    p, dd, h = pd.read_csv(f2), pd.read_csv(f3), pd.read_csv(f4)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.7))
    ax = axes[0]
    xk = np.log2(p.K + 1)
    ax.plot(xk, p.dist, color=RED, lw=2.4, marker="o", label="노드 간 평균 코사인 거리")
    ax.plot(xk, p.acc, color=TEAL, lw=2.4, marker="s", label="Â^K X 로 로지스틱 회귀 (test)")
    ax.set_xticks(xk); ax.set_xticklabels(p.K.astype(int))
    ax.set_xlabel("Â를 곱한 횟수 K (학습 없음)"); ax.set_ylim(0, 1)
    ax.set_title("전파만 반복해도 노드가 같아진다", color=NAVY)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    print(p.round(4))
    ax = axes[1]
    g3 = dd.groupby("layers")[["acc", "dist"]].agg(["mean", "std"])
    L = g3.index.values; xl = np.log2(L)
    ax.errorbar(xl, g3[("acc", "mean")], yerr=g3[("acc", "std")], color=TEAL, lw=2.4, marker="s",
                capsize=3, label="GCN test 정확도")
    ax.axhline(p.acc[0], color=GREY, ls=":", lw=1.2)
    ax.text(np.log2(16), p.acc[0] + 0.02, "특징만 쓴 로지스틱 0.605", ha="right", color="#666", fontsize=9.5)
    ax.set_xticks(xl); ax.set_xticklabels(L)
    ax.set_xlabel("GCN 층 수 (은닉 64)"); ax.set_ylim(0, 1); ax.set_ylabel("test 정확도")
    ax.set_title("GCN 층을 쌓으면", color=NAVY)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    print(g3.round(3))
    ax = axes[2]
    for m, c in [("GCN", TEAL), ("GraphSAGE", BLUE), ("MLP", GREY)]:
        s_ = h[h.model == m].groupby("frac")[["homophily", "acc"]].agg(["mean", "std"])
        ax.errorbar(s_[("homophily", "mean")], s_[("acc", "mean")], yerr=s_[("acc", "std")], color=c,
                    lw=2.4, marker="o", capsize=3, label=m + (" (그래프 안 씀)" if m == "MLP" else ""))
        print(m, s_.round(3))
    ax.invert_xaxis()
    ax.set_xlabel("엣지 homophily (이웃이 같은 분야인 비율)"); ax.set_ylabel("test 정확도")
    ax.set_title("이웃이 서로 다르면", color=NAVY)
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "limits.png"))
    print("done")

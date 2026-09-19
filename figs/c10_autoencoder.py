"""10 §2 — 오토인코더 잠재 공간 vs PCA: 그림이 아니라 숫자로 (PBMC 3k).

입력: 세포 2,638개 x 유전자 1,838개 (scanpy 전처리 완료본, 표준화된 값)
잠재 차원 10으로 통일:
  - PCA (sklearn)
  - 선형 AE: 1838 -> 256 -> 10 -> 256 -> 1838, 활성함수 없음
  - AE:      같은 구조 + ReLU
  - VAE:     같은 구조, 잠재는 평균 mu 사용 (가우시안 복원 + KL)
학습: Adam lr 1e-3, 미니배치 128, 학습 세포의 10%로 early stopping (patience 20)

측정 (세포유형 라벨 8개는 obs['louvain'])
  - kNN 일치도: 잠재 공간 최근접 이웃 15개 중 같은 세포유형 비율
  - Leiden ARI: 잠재 공간 kNN(15) 그래프, resolution 1.0 -> 라벨과의 ARI
  - 선형 probe: 잠재 10차원으로 로지스틱 회귀 5-fold balanced accuracy
  - 복원 오차: 세포 80%로 학습, 나머지 20%의 평균제곱오차
  - 선형 AE와 PCA의 정준상관 (같은 부분공간이면 1)
시드 5개. 결과: figs/cache/c10_ae_metrics.csv, c10_ae_umap.csv
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import pandas as pd
import torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import adjusted_rand_score
from sklearn.cross_decomposition import CCA
from c10_data import load_pbmc, CACHE

torch.set_num_threads(2)
D = 10


class AE(nn.Module):
    def __init__(self, p, act=True, vae=False, d=D, h=256):
        super().__init__()
        A = nn.ReLU if act else nn.Identity
        self.vae = vae
        self.enc = nn.Sequential(nn.Linear(p, h), A())
        self.mu = nn.Linear(h, d)
        self.logvar = nn.Linear(h, d) if vae else None
        self.dec = nn.Sequential(nn.Linear(d, h), A(), nn.Linear(h, p))

    def encode(self, x):
        return self.mu(self.enc(x))

    def forward(self, x):
        h = self.enc(x); mu = self.mu(h)
        if not self.vae:
            return self.dec(mu), 0.0
        lv = self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)       # 재매개변수화
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1).mean()
        return self.dec(z), kl


def fit_ae(X, seed, act=True, vae=False, lr=1e-3, bs=128, max_epochs=500, patience=20):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    i = rng.permutation(len(X)); nv = len(X) // 10
    Xt, Xv = torch.tensor(X[i[nv:]]), torch.tensor(X[i[:nv]])
    m = AE(X.shape[1], act, vae)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    def loss(xb):
        r, kl = m(xb)
        if vae:   # 가우시안 복원(분산 1)의 음의 로그우도 + KL
            return 0.5 * ((r - xb) ** 2).sum(1).mean() + kl
        return ((r - xb) ** 2).mean()

    best, state, wait = np.inf, None, 0
    for ep in range(max_epochs):
        m.train()
        for b in torch.randperm(len(Xt), generator=g).split(bs):
            opt.zero_grad(); loss(Xt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = loss(Xv).item()
        if v < best - 1e-6:
            best, wait, state = v, 0, {k: t.clone() for k, t in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(state); m.eval()
    return m, ep + 1


def embed(m, X):
    with torch.no_grad():
        return m.encode(torch.tensor(X)).numpy()


def recon_mse(m, X):
    with torch.no_grad():
        return float(((m.dec(m.encode(torch.tensor(X))) - torch.tensor(X)) ** 2).mean())


def knn_agree(Z, lab, k=15):
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    idx = nn_.kneighbors(Z, return_distance=False)[:, 1:]
    return float((lab[idx] == lab[:, None]).mean())


def leiden_ari(Z, lab, k=15, res=1.0, umap=False):
    import anndata as ad, scanpy as sc
    a = ad.AnnData(np.zeros((len(Z), 1), np.float32)); a.obsm["Z"] = Z
    sc.pp.neighbors(a, n_neighbors=k, use_rep="Z", random_state=0)
    sc.tl.leiden(a, resolution=res, flavor="igraph", n_iterations=2, random_state=0)
    cl = a.obs["leiden"].values
    out = [adjusted_rand_score(lab, cl), len(set(cl))]
    if umap:
        sc.tl.umap(a, random_state=0)
        out.append(a.obsm["X_umap"])
    return out


def probe(Z, lab):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return float(cross_val_score(clf, Z, lab, cv=cv, scoring="balanced_accuracy").mean())


def cca_mean(A, B):
    a, b = CCA(n_components=D, max_iter=2000).fit_transform(A, B)
    return float(np.mean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(D)]))


if __name__ == "__main__":
    f1 = os.path.join(CACHE, "c10_ae_metrics.csv")
    f2 = os.path.join(CACHE, "c10_ae_umap.csv")
    if not (os.path.exists(f1) and os.path.exists(f2)):
        X, lab, _ = load_pbmc()
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(X)); nte = len(X) // 5
        te, tr = perm[:nte], perm[nte:]
        rows, umaps = [], {}
        Zp = PCA(D, random_state=0).fit_transform(X)
        pca_tr = PCA(D, random_state=0).fit(X[tr])
        rec = float(((pca_tr.inverse_transform(pca_tr.transform(X[te])) - X[te]) ** 2).mean())
        ari, ncl, U = leiden_ari(Zp, lab, umap=True); umaps["PCA"] = U
        rows.append(dict(model="PCA", seed=0, knn=knn_agree(Zp, lab), ari=ari, ncl=ncl,
                         probe=probe(Zp, lab), recon=rec, epochs=np.nan, cca_pca=1.0))
        print(rows[-1], flush=True)
        for name, act, vae in [("선형 AE", False, False), ("AE", True, False), ("VAE", True, True)]:
            for seed in range(5):
                m, ep = fit_ae(X, seed, act, vae)                  # 전체 세포로 학습 → 잠재 공간 평가
                Z = embed(m, X)
                m2, _ = fit_ae(X[tr], seed, act, vae)             # 80%로 학습 → 20% 복원 오차
                want_umap = seed in (0, 1) and name == "AE"
                r = leiden_ari(Z, lab, umap=want_umap)
                if want_umap:
                    umaps[f"AE_s{seed}"] = r[2]
                rows.append(dict(model=name, seed=seed, knn=knn_agree(Z, lab), ari=r[0], ncl=r[1],
                                 probe=probe(Z, lab), recon=recon_mse(m2, X[te]), epochs=ep,
                                 cca_pca=cca_mean(Z, Zp)))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(f1, index=False)
        U = pd.DataFrame({f"{k}_{j}": v[:, j] for k, v in umaps.items() for j in range(2)})
        U["label"] = lab
        U.to_csv(f2, index=False)

    # ── 그림 ──────────────────────────────────────────────────────────
    d, U = pd.read_csv(f1), pd.read_csv(f2)
    g = d.groupby("model", sort=False)[["knn", "ari", "probe", "recon", "cca_pca", "ncl", "epochs"]]
    print(g.mean().round(3)); print(g.std().round(3))
    OUT = outdir("c10")
    types = sorted(U.label.unique())
    pal = [TEAL, PURPLE, ORANGE, NAVY, RED, OLIVE, BLUE, "#8C6D5A"]
    fig = plt.figure(figsize=(17, 5.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.22)
    for k, (key, title) in enumerate([("PCA", "PCA"), ("AE_s0", "오토인코더 (시드 0)"),
                                      ("AE_s1", "오토인코더 (시드 1)")]):
        ax = fig.add_subplot(gs[k])
        for t, c in zip(types, pal):
            m = U.label == t
            ax.scatter(U.loc[m, f"{key}_0"], U.loc[m, f"{key}_1"], s=3, color=c, label=t, rasterized=True)
        row = d[(d.model == ("PCA" if key == "PCA" else "AE")) & (d.seed == (1 if key == "AE_s1" else 0))].iloc[0]
        ax.set_title(f"{title}\nkNN 일치도 {row.knn:.3f} · ARI {row.ari:.2f}", color=NAVY, fontsize=11.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="datalim")
        for sp in ax.spines.values():
            sp.set_visible(False)
        if k == 0:
            ax.legend(frameon=False, fontsize=9, markerscale=3, loc="upper left",
                      bbox_to_anchor=(0.0, -0.02), ncol=4)
    ax = fig.add_subplot(gs[3])
    models = ["PCA", "선형 AE", "AE", "VAE"]
    cols = [TEAL, OLIVE, PURPLE, ORANGE]
    metrics = [("knn", "kNN 일치도"), ("probe", "선형 probe"), ("ari", "Leiden ARI")]
    w = 0.2
    for j, (mname, c) in enumerate(zip(models, cols)):
        sub = d[d.model == mname]
        mu = [sub[m].mean() for m, _ in metrics]; sd = [sub[m].std() if len(sub) > 1 else 0 for m, _ in metrics]
        xs = np.arange(len(metrics)) + (j - 1.5) * w
        ax.bar(xs, mu, w, yerr=sd, color=c, label=mname, capsize=2)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels([t for _, t in metrics])
    ax.set_ylim(0, 1.0); ax.set_ylabel("점수 (1이 최고)")
    ax.set_title("같은 잠재 공간, 숫자로 (시드 5개)", color=NAVY)
    ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.savefig(os.path.join(OUT, "latent.png"), bbox_inches="tight")
    print("wrote", os.path.join(OUT, "latent.png"))

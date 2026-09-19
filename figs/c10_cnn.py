"""10 §1 — CNN은 서열 위를 미끄러지는 모티프 스캐너인가 (합성 DNA, 실측).

길이 200 무작위 서열(ACGT 균등)에서 양성에만 모티프 TGACTCA를 1개 심는다
(위치마다 0.9 확률로 consensus 염기, 아니면 나머지 셋 중 하나).
음성에는 같은 7개 염기를 뒤섞은 서열을 1개 심어 염기 조성 차이를 없앤다.

모델 (모두 학습 데이터의 20%를 validation으로 씀)
  - 정답 PWM 스캔: 심은 모티프를 알고 max log-odds로 판정 (도달 가능한 상한 역할)
  - 1D CNN: 필터 16개 x 폭 12, ReLU, 전역 max pooling, 선형 출력 (파라미터 801)
  - MLP: one-hot 800 -> 64 -> 1 (파라미터 51,329)
  - one-hot 로지스틱 회귀 (파라미터 801, C는 validation으로)
  - 6-mer 빈도 로지스틱 회귀 (특징 4,096, C는 validation으로)

실험 1: 학습 서열 수 100-10,000, test 2,000, 시드 3
실험 2: 학습 3,000개에서는 모티프를 늘 50번 위치에만 심고, test는 같은 위치(50) / 다른 위치(150)
실험 3: 학습된 CNN 필터와 심은 모티프 비교
결과는 figs/cache/c10_cnn_*.csv 에 저장 (첫 실행 수십 분)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import pandas as pd
import torch, torch.nn as nn
from scipy import sparse
from sklearn.linear_model import LogisticRegression

torch.set_num_threads(2)
OUT = outdir("c10")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
L, MOTIF, P_CONS = 200, "TGACTCA", 0.90
BASES = "ACGT"
M_IDX = np.array([BASES.index(b) for b in MOTIF])
K = len(MOTIF)


def make(n, rng, lo=0, hi=L - K):
    """n개 서열(절반 양성). 반환: 정수 서열 (n, L), 라벨, 모티프 시작 위치"""
    S = rng.integers(0, 4, (n, L))
    y = np.zeros(n, int); y[: n // 2] = 1
    pos = rng.integers(lo, hi + 1, n)
    for i in range(n):
        if y[i]:
            m = M_IDX.copy()
            mut = rng.random(K) > P_CONS
            m[mut] = (m[mut] + rng.integers(1, 4, mut.sum())) % 4   # 나머지 셋 중 하나
        else:
            m = M_IDX.copy()
            while (m == M_IDX).all():
                m = rng.permutation(M_IDX)                          # 같은 염기, 순서만 섞음
        S[i, pos[i]: pos[i] + K] = m
    perm = rng.permutation(n)
    return S[perm], y[perm], pos[perm]


def onehot(S):
    return np.eye(4, dtype=np.float32)[S].transpose(0, 2, 1)       # (n, 4, L)


def kmer(S, k=6):
    w = 4 ** np.arange(k - 1, -1, -1)
    idx = sum(S[:, j: L - k + 1 + j] * w[j] for j in range(k))      # (n, L-k+1)
    rows = np.repeat(np.arange(len(S)), idx.shape[1])
    return sparse.csr_matrix((np.ones(idx.size, np.float32), (rows, idx.ravel())),
                             shape=(len(S), 4 ** k))


class CNN(nn.Module):
    def __init__(self, n_filters=16, width=12):
        super().__init__()
        self.conv = nn.Conv1d(4, n_filters, width)
        self.fc = nn.Linear(n_filters, 1)

    def forward(self, x):
        h = torch.relu(self.conv(x))          # (n, 필터, 위치)
        return self.fc(h.max(dim=2).values).squeeze(1)   # 전역 max pooling


def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(4 * L, 64), nn.ReLU(), nn.Linear(64, 1),
                         nn.Flatten(0))


def fit_nn(model, X, y, seed, lr=1e-3, bs=64, max_epochs=1000, patience=100):
    rng = np.random.default_rng(seed)
    i = rng.permutation(len(y)); nv = len(y) // 5
    va, tr = i[:nv], i[nv:]
    Xt, yt = torch.tensor(X[tr]), torch.tensor(y[tr], dtype=torch.float32)
    Xv, yv = torch.tensor(X[va]), torch.tensor(y[va], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = nn.BCEWithLogitsLoss()
    best, state, wait = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        model.train()
        for b in torch.randperm(len(yt), generator=g).split(bs):
            opt.zero_grad(); lf(model(Xt[b]), yt[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = lf(model(Xv), yv).item()
        if v < best - 1e-5:
            best, wait = v, 0
            state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model


def fit_lr(F, y, seed):
    rng = np.random.default_rng(seed)
    i = rng.permutation(len(y)); nv = len(y) // 5
    va, tr = i[:nv], i[nv:]
    best = max((LogisticRegression(C=C, max_iter=3000).fit(F[tr], y[tr]) for C in (0.01, 0.1, 1.0)),
               key=lambda m: m.score(F[va], y[va]))
    return LogisticRegression(C=best.C, max_iter=3000).fit(F, y)


def acc_nn(model, X, y):
    with torch.no_grad():
        return float(((model(torch.tensor(X)) > 0).numpy() == y).mean())


def pwm_scan(S):
    """정답 모티프를 아는 모형: 위치마다 PWM log-odds 합, 그중 최댓값"""
    P = np.full((4, K), (1 - P_CONS) / 3); P[M_IDX, np.arange(K)] = P_CONS
    LO = np.log(P / 0.25)
    return np.stack([LO[S[:, j: j + K], np.arange(K)].sum(1) for j in range(L - K + 1)], 1).max(1)


def run_all(Str, ytr, Ste, yte, seed):
    out = {}
    s_tr, s_te = pwm_scan(Str), pwm_scan(Ste)       # 임계값만 학습 데이터로 정함
    ts = np.unique(s_tr)
    t = ts[np.argmax([((s_tr > t) == ytr).mean() for t in ts])]
    out["정답 PWM 스캔"] = float(((s_te > t) == yte).mean())
    torch.manual_seed(seed); m = fit_nn(CNN(), onehot(Str), ytr, seed)
    out["CNN"] = acc_nn(m, onehot(Ste), yte); cnn = m
    torch.manual_seed(seed); m = fit_nn(mlp(), onehot(Str), ytr, seed)
    out["MLP"] = acc_nn(m, onehot(Ste), yte)
    flat = lambda S: onehot(S).reshape(len(S), -1)
    out["one-hot 로지스틱"] = fit_lr(flat(Str), ytr, seed).score(flat(Ste), yte)
    out["6-mer 로지스틱"] = fit_lr(kmer(Str), ytr, seed).score(kmer(Ste), yte)
    return out, cnn


if __name__ == "__main__":
    f1 = os.path.join(CACHE, "c10_cnn_curve.csv")
    f2 = os.path.join(CACHE, "c10_cnn_shift.csv")
    f3 = os.path.join(CACHE, "c10_cnn_filter.csv")
    if not os.path.exists(f1):
        rows = []
        for seed in range(3):
            rng = np.random.default_rng(100 + seed)
            Ste, yte, _ = make(2000, rng)
            for n in [100, 300, 1000, 3000, 10000]:
                Str, ytr, _ = make(n, rng)
                t = time.time()
                res, cnn = run_all(Str, ytr, Ste, yte, seed)
                rows += [dict(seed=seed, n=n, model=k, acc=v) for k, v in res.items()]
                print(seed, n, {k: round(v, 3) for k, v in res.items()}, f"{time.time()-t:.0f}s", flush=True)
                if seed == 0 and n == 3000:
                    W = cnn.conv.weight.detach().numpy()          # (16, 4, 12)
                    np.savetxt(f3, W.reshape(16, -1), delimiter=",")
                    torch.save(cnn.state_dict(), os.path.join(CACHE, "c10_cnn_seed0.pt"))
        pd.DataFrame(rows).to_csv(f1, index=False)
    if not os.path.exists(f2):
        rows = []
        for seed in range(3):
            rng = np.random.default_rng(200 + seed)
            Str, ytr, _ = make(3000, rng, 50, 50)
            Ste, yte, _ = make(2000, rng, 150, 150)
            Ssame, ysame, _ = make(2000, rng, 50, 50)
            res, _ = run_all(Str, ytr, Ste, yte, seed)
            res2, _ = run_all(Str, ytr, Ssame, ysame, seed)
            rows += [dict(seed=seed, model=k, test="다른 위치 (150)", acc=v) for k, v in res.items()]
            rows += [dict(seed=seed, model=k, test="같은 위치 (50)", acc=v) for k, v in res2.items()]
            print("shift", seed, res, res2, flush=True)
        pd.DataFrame(rows).to_csv(f2, index=False)

    # ── 그림 ──────────────────────────────────────────────────────────
    d1, d2 = pd.read_csv(f1), pd.read_csv(f2)
    W = np.loadtxt(f3, delimiter=",").reshape(16, 4, 12)
    P = np.full((4, K), (1 - P_CONS) / 3); P[M_IDX, np.arange(K)] = P_CONS
    LO = np.log(P / 0.25)
    corr = np.array([[np.corrcoef(W[f][:, o:o + K].ravel(), LO.ravel())[0, 1]
                      for o in range(12 - K + 1)] for f in range(16)])
    fb = int(corr.max(1).argmax()); ob = int(corr[fb].argmax())
    print("필터별 최대 상관:", np.round(np.sort(corr.max(1))[::-1], 3))
    rng = np.random.default_rng(100); Ste, yte, pos = make(2000, rng)
    Xo = onehot(Ste)
    act = np.stack([(Xo[:, :, j: j + 12] * W[fb]).sum((1, 2)) for j in range(L - 12 + 1)], 1)
    hit = (act.argmax(1) + ob == pos)[yte == 1].mean()
    print(f"최고 필터 {fb}: 상관 {corr[fb, ob]:.3f}, 양성 test 서열에서 최대 반응 위치 = 심은 위치 {hit:.3f}")

    MOD = [("정답 PWM 스캔", NAVY, "--"), ("CNN", TEAL, "-"), ("6-mer 로지스틱", ORANGE, "-"),
           ("MLP", PURPLE, "-"), ("one-hot 로지스틱", GREY, "-")]
    fig = plt.figure(figsize=(16, 4.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 0.9], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    g = d1.groupby(["model", "n"])["acc"].agg(["mean", "std"]).reset_index()
    for m, c, ls in MOD:
        s_ = g[g.model == m]
        ax.errorbar(s_.n, s_["mean"], yerr=s_["std"], color=c, ls=ls, lw=2.2, marker="o", ms=5,
                    capsize=3, label=m)
        print(m, dict(zip(s_.n, s_["mean"].round(3))))
    ax.axhline(0.5, color=GREY, lw=1, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0.44, 0.87)
    ax.set_xlabel("학습 서열 수"); ax.set_ylabel("test 정확도")
    ax.set_title("모티프가 아무 위치에나 있을 때", color=NAVY)
    ax.legend(frameon=False, fontsize=9.5, loc="center right")

    ax = fig.add_subplot(gs[:, 1])
    g2 = d2.groupby(["model", "test"])["acc"].mean().unstack()
    names = [m for m, _, _ in MOD]
    xs = np.arange(len(names)); w = 0.38
    for k, (t, c) in enumerate([("같은 위치 (50)", OLIVE), ("다른 위치 (150)", RED)]):
        v = g2.loc[names, t].values
        ax.bar(xs + (k - 0.5) * w, v, w, color=c, label=f"test: {t}")
        for x_, y_ in zip(xs + (k - 0.5) * w, v):
            ax.text(x_, y_ + 0.01, f"{y_:.2f}", ha="center", fontsize=8.5)
    print(g2.round(3))
    ax.axhline(0.5, color=GREY, lw=1, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax.set_ylim(0.4, 1.1); ax.set_ylabel("test 정확도")
    ax.set_title("학습 때 모티프가 늘 50번 위치였다면", color=NAVY)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    for r, (M, title) in enumerate([(LO, "심은 모티프 (log-odds)"),
                                    (W[fb][:, ob: ob + K], f"CNN이 배운 필터 (상관 {corr[fb, ob]:.2f})")]):
        ax = fig.add_subplot(gs[r, 2])
        v = np.abs(M).max()
        ax.imshow(M, cmap="PuOr_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_yticks(range(4)); ax.set_yticklabels(list(BASES))
        ax.set_xticks(range(K)); ax.set_xticklabels(list(MOTIF) if r == 0 else [""] * K)
        ax.set_title(title, color=NAVY, fontsize=11.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cnn_motif.png"))
    print("wrote", os.path.join(OUT, "cnn_motif.png"))

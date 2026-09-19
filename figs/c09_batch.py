"""09 §4 — 라벨과 섞인 batch는 지름길이 된다 (합성 batch, 실제 발현).

LumA vs LumB 실제 발현에 기술적 인공물을 심는다: batch 2 샘플은 무작위 유전자 200개(전체 1%)의
log2 발현에 +delta. 학습 240명(LumA 120 + LumB 120)에서는 LumB의 90%, LumA의 10%가 batch 2.
평가: 나머지 456명을 (a) batch 50% 무작위 (b) 방향을 뒤집어 LumA 전부 batch 2, LumB 전부 batch 1.
모델은 c09_models 의 fit_l2 / fit_rf / fit_mlp (검증셋으로 튜닝). 반복 5회.
결과는 figs/cache/c09_batch.csv (첫 실행 약 40분).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from c09_models import *
from c09_data import CACHE

OUT = outdir("c09")
f = os.path.join(CACHE, "c09_batch.csv")
if not os.path.exists(f):
    X, y = load()
    keep = y.isin(["LumA", "LumB"]); X = X[keep]; yb = (y[keep] == "LumB").astype(int).values
    rows = []
    for r in range(5):
        rng = np.random.default_rng(r)
        genes = rng.choice(X.shape[1], 200, replace=False)
        tr = np.r_[rng.choice(np.where(yb == 0)[0], 120, replace=False),
                   rng.choice(np.where(yb == 1)[0], 120, replace=False)]
        te = np.setdiff1d(np.arange(len(X)), tr)
        b_tr = np.where(yb[tr] == 1, rng.random(len(tr)) < 0.9, rng.random(len(tr)) < 0.1)
        b_rand = rng.random(len(te)) < 0.5
        b_flip = yb[te] == 0
        for delta in [0.0, 0.5, 1.0, 2.0]:
            def add(Xs, b):
                Z = Xs.copy(); Z.iloc[np.where(b)[0], genes] += delta; return Z
            Xtr = add(X.iloc[tr], b_tr)
            Xa, Xv, ya, yv, ba, bv = train_test_split(Xtr, yb[tr], b_tr, test_size=0.2,
                                                      stratify=yb[tr], random_state=r)
            A, V = prep(Xa, Xv)
            A_full, T_rand = prep(Xtr, add(X.iloc[te], b_rand))
            _, T_flip = prep(Xtr, add(X.iloc[te], b_flip))
            row = {"rep": r, "delta": delta}
            for name, fit in [("L2", fit_l2), ("RF", fit_rf), ("MLP", fit_mlp)]:
                m, _ = fit(A, ya, V, yv, A_full, yb[tr])
                row[name + "_rand"] = bacc(yb[te], m.predict(T_rand))
                row[name + "_flip"] = bacc(yb[te], m.predict(T_flip))
            rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(f, index=False)
d = pd.read_csv(f)
g = d.groupby("delta")
print(g[[c for c in d.columns if c.endswith(("_rand", "_flip"))]].mean().round(3).to_string())

fig, ax = plt.subplots(figsize=(8.8, 4.8))
xs = np.array(sorted(d.delta.unique()))
for name, c, lab in [("L2", NAVY, "로지스틱 회귀 (L2)"), ("RF", ORANGE, "랜덤 포레스트"),
                     ("MLP", TEAL, "MLP")]:
    m, s = g[name + "_flip"].mean(), g[name + "_flip"].std()
    ax.errorbar(xs, m, yerr=s, color=c, lw=2.2, marker="o", ms=6, capsize=3, label=lab)
    ax.plot(xs, g[name + "_rand"].mean(), color=c, lw=1.3, ls=":", alpha=0.9)
    ax.text(xs[-1] + 0.06, m.iloc[-1], f"{m.iloc[-1]:.2f}", color=c, va="center", fontsize=10)
ax.axhline(0.5, color=GREY, ls="--", lw=1)
ax.text(1.35, 0.508, "우연 (0.5)", color="#777", fontsize=9.5)
ax.set_xticks(xs); ax.set_xticklabels([f"+{v:g}\n({2 ** v:.1f}배)" for v in xs])
ax.set_xlim(-0.15, 2.35); ax.set_ylim(0.25, 0.95)
ax.set_xlabel("batch 2에서 유전자 200개에 더한 값 (log2)")
ax.set_ylabel("balanced accuracy")
ax.set_title("학습 때 batch와 라벨이 섞였고, test에서 방향이 뒤집히면", color=NAVY)
ax.legend(frameon=False, loc="lower left", title="실선: 방향을 뒤집은 test · 점선: batch 무작위 test",
          title_fontsize=9, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "batch.png"))
print("wrote", os.path.join(OUT, "batch.png"))

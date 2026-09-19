"""09 §1 — 누수 두 가지를 무작위 라벨로 실측.

무작위 라벨이므로 어떤 방법이든 진짜 정확도는 0.5.
왼쪽: 특징 선택(상위 50 유전자, F-검정)을 CV 밖(전체 샘플)에서 했을 때 vs 안에서 했을 때.
      TCGA BRCA 발현에서 환자 100명을 뽑아 20,511 유전자, 반복 50회.
오른쪽: 후보 모델 200개(무작위 유전자 50개 × L2 로지스틱)를 같은 test 100명으로 평가해
      최고를 골랐을 때의 test 정확도와, 그 모델의 새 데이터(500명) 정확도. 반복 20회.
결과는 figs/cache/c09_leak*.csv 에 저장하고, 있으면 다시 계산하지 않습니다 (첫 실행 약 10분).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import pandas as pd, warnings
from c09_data import load_brca, CACHE
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

OUT = outdir("c09")
X = load_brca()[0].values.astype(np.float32)

# ── 실험 1: 특징 선택 위치 ─────────────────────────────────────────────
f1 = os.path.join(CACHE, "c09_leak_select.csv")
if not os.path.exists(f1):
    K, N, rows = 50, 100, []
    for r in range(50):
        rng = np.random.default_rng(r)
        Xs = X[rng.choice(len(X), N, replace=False)]
        Xs = Xs[:, Xs.std(0) > 0]
        y = rng.permutation(np.repeat([0, 1], N // 2))
        top = np.argsort(-np.nan_to_num(f_classif(Xs, y)[0]))[:K]
        aw, ar = [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(Xs, y):
            m = LogisticRegression(max_iter=2000).fit(Xs[tr][:, top], y[tr])
            aw.append(m.score(Xs[te][:, top], y[te]))
            ti = np.argsort(-np.nan_to_num(f_classif(Xs[tr], y[tr])[0]))[:K]
            m = LogisticRegression(max_iter=2000).fit(Xs[tr][:, ti], y[tr])
            ar.append(m.score(Xs[te][:, ti], y[te]))
        rows.append({"wrong": np.mean(aw), "right": np.mean(ar)})
    pd.DataFrame(rows).to_csv(f1, index=False)
sel = pd.read_csv(f1)

# ── 실험 2: test 재사용 ──────────────────────────────────────────────
f2, f2b = os.path.join(CACHE, "c09_leak_reuse.csv"), os.path.join(CACHE, "c09_leak_reuse_rep0.csv")
if not os.path.exists(f2):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    rows = []
    for r in range(20):
        rng = np.random.default_rng(100 + r)
        idx = rng.permutation(len(Z)); tr, te, fr = idx[:100], idx[100:200], idx[200:700]
        y = rng.integers(0, 2, len(Z))
        acc = []
        for _ in range(200):
            g = rng.choice(Z.shape[1], 50, replace=False)
            m = LogisticRegression(C=0.1, max_iter=2000).fit(Z[tr][:, g], y[tr])
            acc.append((m.score(Z[te][:, g], y[te]), m.score(Z[fr][:, g], y[fr])))
        acc = np.array(acc); b = acc[:, 0].argmax()
        rows.append({"test_mean": acc[:, 0].mean(), "best_test": acc[b, 0], "best_fresh": acc[b, 1]})
        if r == 0:
            pd.DataFrame(acc, columns=["test", "fresh"]).to_csv(f2b, index=False)
    pd.DataFrame(rows).to_csv(f2, index=False)
reuse, rep0 = pd.read_csv(f2), pd.read_csv(f2b)

print("특징 선택 CV 밖:", sel.wrong.mean().round(3), "±", sel.wrong.std().round(3),
      "| CV 안:", sel.right.mean().round(3), "±", sel.right.std().round(3))
print("test 재사용:", reuse.mean().round(3).to_dict(), "sd", reuse.std().round(3).to_dict())

# ── 그림 ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
rng = np.random.default_rng(0)
LABS = ["전체 샘플로 유전자 선택\n→ 그다음 CV", "CV의 학습 fold 안에서\n유전자 선택"]
for i, (col, c) in enumerate([("wrong", RED), ("right", TEAL)]):
    v = sel[col].values
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=16, color=c, alpha=0.75, lw=0)
    ax.hlines(v.mean(), i - 0.25, i + 0.25, color=NAVY, lw=2.2)
    ax.text(i + 0.29, v.mean(), f"{v.mean():.2f}", va="center", color=NAVY, fontsize=11,
            fontweight="bold")
ax.axhline(0.5, color=GREY, ls="--", lw=1.2)
ax.text(-0.45, 0.507, "진짜 정확도 0.5", color="#777", fontsize=9.5, ha="left", va="bottom")
ax.set_xticks([0, 1]); ax.set_xticklabels(LABS, fontsize=10)
ax.set_xlim(-0.5, 1.6); ax.set_ylim(0.3, 0.95)
ax.set_ylabel("5-fold CV 정확도")
ax.set_title("무작위 라벨인데 0.77이 나온다", color=NAVY)

ax = axes[1]
ax.hist(rep0.test, bins=np.arange(0.3, 0.72, 0.02), color=LGREY, edgecolor="white")
b = rep0.test.idxmax()
ax.axvline(rep0.test[b], color=RED, lw=2)
ax.axvline(rep0.fresh[b], color=TEAL, lw=2, ls="--")
ymax = ax.get_ylim()[1]
ax.annotate("", xy=(rep0.fresh[b], ymax * 0.82), xytext=(rep0.test[b], ymax * 0.82),
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.6))
ax.text(rep0.test[b] - 0.006, ymax * 0.93, f"test로 고른 최고 {rep0.test[b]:.2f}", color=RED,
        ha="right", fontsize=10)
ax.text(0.305, ymax * 0.62, f"그 모델을 새 데이터\n500명에 적용하면 {rep0.fresh[b]:.2f}",
        color=TEAL, ha="left", fontsize=10)
ax.set_xlabel("test 정확도 (후보 모델 200개)")
ax.set_ylabel("모델 수")
ax.set_title(f"test를 200번 보면 — 20회 평균: {reuse.best_test.mean():.2f} → {reuse.best_fresh.mean():.2f}",
             color=NAVY)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "leakage.png"))
print("wrote", os.path.join(OUT, "leakage.png"))

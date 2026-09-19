"""09 §2 — MLP 학습 곡선: train loss는 계속 내려가고 validation loss는 어느 순간 올라간다.

TCGA BRCA PAM50 5-class, 학습 588명 / 검증 147명 (split random_state=0),
고분산 유전자 2,000개, 2000 → 256 → 5 MLP, AdamW lr=1e-4, 미니배치 64, 300 epoch 끝까지.
두 설정: 정규화 없음 (dropout 0, weight decay 0) / dropout 0.5 + weight decay 5.
결과는 figs/cache/c09_curves.csv 에 저장 (첫 실행 약 3분).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import pandas as pd
from c09_models import load, prep, train_mlp, train_test_split
from c09_data import CACHE

OUT = outdir("c09")
f = os.path.join(CACHE, "c09_curves.csv")
if not os.path.exists(f):
    X, y = load()
    classes = sorted(y.unique()); yi = y.map({c: i for i, c in enumerate(classes)}).values
    Xtr, Xte, ytr, yte = train_test_split(X, yi, test_size=0.25, stratify=yi, random_state=0)
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=0)
    A, V = prep(Xa, Xv)
    cols = {}
    for name, wd, dp in [("none", 0.0, 0.0), ("reg", 5.0, 0.5)]:
        _, _, hist = train_mlp(A, ya, V, yv, lr=1e-4, wd=wd, dropout=dp, max_epochs=300,
                               patience=10**9, history=True)
        h = np.array(hist); cols[name + "_train"] = h[:, 0]; cols[name + "_val"] = h[:, 1]
    pd.DataFrame(cols).to_csv(f, index=False)
d = pd.read_csv(f)
ep = np.arange(1, len(d) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
SET = [("none", RED, "정규화 없음"), ("reg", TEAL, "dropout 0.5 + weight decay 5")]
ax = axes[0]
for name, c, lab in SET:
    ax.plot(ep, d[name + "_train"], color=c, lw=1.6, alpha=0.55, label=f"{lab} — train")
    ax.plot(ep, d[name + "_val"], color=c, lw=2.4, label=f"{lab} — validation")
ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss (로그 축)")
ax.set_title("train은 0으로 가고, validation은 멈춘다", color=NAVY)
ax.legend(frameon=False, fontsize=9.5, loc="lower left")

ax = axes[1]
for name, c, lab in SET:
    tr, va = d[name + "_train"], d[name + "_val"]
    b = int(va.values.argmin())
    ax.plot(ep, va, color=c, lw=2.2, label=lab)
    ax.scatter([b + 1], [va[b]], color=c, s=60, zorder=5, edgecolor="white", lw=1.2)
    ax.text(302, va.iloc[-1], f"{va.iloc[-1]:.3f}", color=c, va="center", fontsize=10)
    print(f"{lab}: val 최저 epoch {b + 1}, val {va[b]:.4f} | 300 epoch: val {va.iloc[-1]:.4f}, "
          f"train {tr.iloc[-1]:.5f}")
bn = int(d["none_val"].values.argmin())
ax.annotate(f"최저점 epoch {bn + 1}, {d['none_val'][bn]:.3f}\n= early stopping 지점",
            xy=(bn + 1, d["none_val"][bn]), xytext=(bn + 40, 0.245), color=NAVY, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2))
ax.set_ylim(0.23, 0.40); ax.set_xlim(0, 330)
ax.set_xlabel("epoch"); ax.set_ylabel("validation loss")
ax.set_title("validation만 확대하면", color=NAVY)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "curves.png"))
print("wrote", os.path.join(OUT, "curves.png"))

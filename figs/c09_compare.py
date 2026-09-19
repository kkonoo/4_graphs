"""09 §3 — 같은 조건에서 비교: 로지스틱 회귀(L2) · elastic net · 랜덤 포레스트 · MLP.

(1) 전체: TCGA BRCA PAM50 5-class, test 25%(246명) 고정, 나머지 735명으로 학습. split 10회.
(2) 학습 곡선: 같은 test에 대해 학습 샘플을 60 / 120 / 240 / 480명으로 줄임. split 5회.
(3) 시뮬레이션: 특징 500개(표준정규), 참 함수가 선형(logit = 2·s1) 또는 선형 + 상호작용
    (logit = s1 + 2.5·s2·s3). s_k는 특징 5개씩의 합/√5. test 5,000개, 반복 3회.
모든 모델은 학습 데이터의 20%를 validation으로 떼어 하이퍼파라미터를 고른 뒤 전체 학습 데이터로
다시 학습합니다 (c09_models.fit_*). 결과는 figs/cache/c09_{full,lc,sim}.csv.
첫 실행은 수 시간 걸립니다.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from c09_models import *
from c09_data import CACHE

OUT = outdir("c09")
MODELS = [("L2", fit_l2), ("EN", fit_en), ("RF", fit_rf), ("MLP", fit_mlp)]


def brca():
    X, y = load()
    classes = sorted(y.unique())
    return X, y.map({c: i for i, c in enumerate(classes)}).values


f_full = os.path.join(CACHE, "c09_full.csv")
if not os.path.exists(f_full):
    X, yi = brca(); rows = []
    for rep in range(10):
        Xtr, Xte, ytr, yte = train_test_split(X, yi, test_size=0.25, stratify=yi, random_state=rep)
        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=rep)
        A, V = prep(Xa, Xv); A_full, B = prep(Xtr, Xte)
        row = {"rep": rep, "majority": bacc(yte, np.full_like(yte, np.bincount(ytr).argmax()))}
        for name, fit in MODELS:
            m, hp = fit(A, ya, V, yv, A_full, ytr)
            row[name] = bacc(yte, m.predict(B)); row[name + "_hp"] = json.dumps(hp)
        rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(f_full, index=False)

f_lc = os.path.join(CACHE, "c09_lc.csv")
if not os.path.exists(f_lc):
    X, yi = brca(); rows = []
    for rep in range(5):
        Xtr, Xte, ytr, yte = train_test_split(X, yi, test_size=0.25, stratify=yi, random_state=rep)
        for n in [60, 120, 240, 480]:
            Xn, _, yn, _ = train_test_split(Xtr, ytr, train_size=n, stratify=ytr, random_state=rep)
            Xa, Xv, ya, yv = train_test_split(Xn, yn, test_size=0.2, stratify=yn, random_state=rep)
            A, V = prep(Xa, Xv); A_full, B = prep(Xn, Xte)
            row = {"rep": rep, "n": n}
            for name, fit in MODELS:
                m, hp = fit(A, ya, V, yv, A_full, yn)
                row[name] = bacc(yte, m.predict(B)); row[name + "_hp"] = json.dumps(hp)
            rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(f_lc, index=False)

f_sim = os.path.join(CACHE, "c09_sim.csv")
if not os.path.exists(f_sim):
    P = 500

    def make(n, truth, rng):
        Z = rng.normal(size=(n, P)).astype(np.float32)
        s1, s2, s3 = (Z[:, i * 5:(i + 1) * 5].sum(1) / np.sqrt(5) for i in range(3))
        logit = 2.0 * s1 if truth == "linear" else 1.0 * s1 + 2.5 * s2 * s3
        return Z, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64), logit

    rows = []
    for truth in ["linear", "interaction"]:
        for rep in range(3):
            rng = np.random.default_rng(1000 * (truth == "interaction") + rep)
            Zte, yte, lg = make(5000, truth, rng)
            for n in [100, 300, 1000, 3000]:
                Zn, yn, _ = make(n, truth, rng)
                Za, Zv, ya, yv = train_test_split(Zn, yn, test_size=0.2, stratify=yn, random_state=rep)
                row = {"truth": truth, "rep": rep, "n": n, "bayes": float(((lg > 0) == yte).mean())}
                for name, fit in [("L2", fit_l2), ("RF", fit_rf), ("MLP", fit_mlp)]:
                    m, hp = fit(Za, ya, Zv, yv, Zn, yn)
                    row[name] = float((m.predict(Zte) == yte).mean()); row[name + "_hp"] = json.dumps(hp)
                rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(f_sim, index=False)

full, lc, sim = pd.read_csv(f_full), pd.read_csv(f_lc), pd.read_csv(f_sim)
print("전체 (10 split):", {k: f"{full[k].mean():.3f} ± {full[k].std():.3f}"
                          for k in ["majority", "L2", "EN", "RF", "MLP"]})
dd = full.MLP - full.EN
print(f"MLP − EN: {dd.mean():+.3f} ± {dd.std():.3f}, MLP 우세 {(dd > 0).sum()}/{len(dd)}")
print("학습 곡선 평균:\n", lc.groupby("n")[["L2", "EN", "RF", "MLP"]].mean().round(3))
print("시뮬레이션 평균:\n", sim.groupby(["truth", "n"])[["L2", "RF", "MLP", "bayes"]].mean().round(3))

# ── 그림 ───────────────────────────────────────────────────────────────
STY = [("L2", NAVY, "로지스틱 회귀 (L2)"), ("EN", PURPLE, "elastic net"),
       ("RF", ORANGE, "랜덤 포레스트"), ("MLP", TEAL, "MLP")]
fig = plt.figure(figsize=(13.5, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.38, wspace=0.22)
axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
ax = axes[0]
lc_all = pd.concat([lc, full.assign(n=735)[["rep", "n", "L2", "EN", "RF", "MLP"]]])
for k, c, lab in STY:
    g = lc_all.groupby("n")[k]
    ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), color=c, marker="o", ms=5, lw=2,
                capsize=3, label=lab)
ax.axhline(0.2, color=GREY, ls="--", lw=1); ax.text(62, 0.215, "다수 클래스 0.20", color="#777", fontsize=9)
ax.set_xscale("log"); ax.set_xticks([60, 120, 240, 480, 735]); ax.set_xticklabels([60, 120, 240, 480, 735])
ax.set_xlabel("학습 샘플 수 (환자)"); ax.set_ylabel("balanced accuracy (test 246명)")
ax.set_title("실제 데이터: TCGA 유방암 PAM50", color=NAVY)
ax.legend(frameon=False, fontsize=10, loc="lower right", ncol=2)
for ax, truth, title in [(axes[1], "linear", "시뮬레이션: 참 함수가 선형"),
                         (axes[2], "interaction", "시뮬레이션: 선형 + 상호작용")]:
    s = sim[sim.truth == truth]
    for k, c, lab in [x for x in STY if x[0] != "EN"]:
        g = s.groupby("n")[k]
        ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), color=c, marker="o", ms=5, lw=2,
                    capsize=3, label=lab)
    bayes = s.bayes.mean()
    ax.axhline(bayes, color=NAVY, ls=":", lw=1.2)
    ax.text(105, bayes + 0.008, f"참 모형의 정확도 {bayes:.2f}", color=NAVY, fontsize=9)
    ax.set_xscale("log"); ax.set_xticks([100, 300, 1000, 3000]); ax.set_xticklabels([100, 300, "1,000", "3,000"])
    ax.set_xlabel("학습 샘플 수"); ax.set_ylabel("정확도 (test 5,000개)")
    ax.set_ylim(0.45, bayes + 0.06)
    ax.set_title(title, color=NAVY)
axes[1].legend(frameon=False, fontsize=9.5, loc="lower right")
fig.savefig(os.path.join(OUT, "compare.png"))
print("wrote", os.path.join(OUT, "compare.png"))

"""09 §4 — 실제 TCGA 채취 기관(tissue source site, 바코드 6–7번째 글자)의 흔적.

(1) LumA 환자만 골라 5개 기관(BH, A2, E2, A8, D8)을 발현으로 맞히기. 5-fold × 3회.
(2) LumA vs LumB 학습 240명을 기관 그룹과 완전히 섞었을 때(LumA는 그룹1, LumB는 그룹2에서만)와
    섞지 않았을 때(두 그룹 반반)를, 방향을 뒤집은 샘플(LumA∈그룹2, LumB∈그룹1)에서 비교. 기관을
    무작위로 반씩 나누는 것을 10회.
그림 없음 — 수치만 출력. 결과는 figs/cache/c09_site_*.csv (첫 실행 약 1시간).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c09_models import *
from c09_data import CACHE
from sklearn.model_selection import StratifiedKFold

X, y = load()
site = pd.Series(X.index.str[5:7], index=X.index)

# ── (1) 기관 맞히기 ──────────────────────────────────────────────────
f1 = os.path.join(CACHE, "c09_site_predict.csv")
if not os.path.exists(f1):
    keep = (y == "LumA") & site.isin(["BH", "A2", "E2", "D8", "A8"])
    Xs, s = X[keep], site[keep]
    codes = sorted(s.unique()); si = s.map({c: i for i, c in enumerate(codes)}).values
    rows = []
    for rep in range(3):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(Xs, si)):
            Xtr, Xte, ytr, yte = Xs.iloc[tr], Xs.iloc[te], si[tr], si[te]
            Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=rep)
            A, V = prep(Xa, Xv); A_full, B = prep(Xtr, Xte)
            row = {"rep": rep, "fold": k}
            for name, fit in [("L2", fit_l2), ("RF", fit_rf), ("MLP", fit_mlp)]:
                m, _ = fit(A, ya, V, yv, A_full, ytr); row[name] = bacc(yte, m.predict(B))
            rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(f1, index=False)
d1 = pd.read_csv(f1)
print("기관 맞히기 (LumA, 5기관, 우연 0.2):",
      {k: f"{d1[k].mean():.3f} ± {d1[k].std():.3f}" for k in ["L2", "RF", "MLP"]})

# ── (2) 기관 그룹과 라벨을 섞어 학습 ─────────────────────────────────
f2 = os.path.join(CACHE, "c09_site_confound.csv")
if not os.path.exists(f2):
    keep = y.isin(["LumA", "LumB"])
    Xl, yl, sl = X[keep], (y[keep] == "LumB").astype(int).values, site[keep]
    sites = sl.unique(); rows = []
    for r in range(10):
        rng = np.random.default_rng(r)
        G = sl.isin(set(rng.choice(sites, len(sites) // 2, replace=False))).values
        idx = {(lab, grp): rng.permutation(np.where((yl == lab) & (G == grp))[0])
               for lab in (0, 1) for grp in (True, False)}
        n = 60
        conf = np.r_[idx[(0, True)][:2 * n], idx[(1, False)][:2 * n]]
        ctrl = np.r_[idx[(0, True)][2 * n:3 * n], idx[(0, False)][:n],
                     idx[(1, False)][2 * n:3 * n], idx[(1, True)][:n]]
        used = set(conf) | set(ctrl)
        anti = np.array([i for i in np.r_[idx[(0, False)], idx[(1, True)]] if i not in used], dtype=int)
        row = {"rep": r, "n_anti": len(anti)}
        for design, tr in [("conf", conf), ("ctrl", ctrl)]:
            Xtr, ytr = Xl.iloc[tr], yl[tr]
            Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=r)
            A, V = prep(Xa, Xv); A_full, B = prep(Xtr, Xl.iloc[anti])
            for name, fit in [("L2", fit_l2), ("MLP", fit_mlp)]:
                m, _ = fit(A, ya, V, yv, A_full, ytr)
                row[f"{design}_{name}"] = bacc(yl[anti], m.predict(B))
        rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(f2, index=False)
d2 = pd.read_csv(f2)
print("기관과 섞어 학습 → 방향을 뒤집은 샘플에서 (10회 평균):",
      {k: round(d2[k].mean(), 3) for k in ["conf_L2", "ctrl_L2", "conf_MLP", "ctrl_MLP"]})

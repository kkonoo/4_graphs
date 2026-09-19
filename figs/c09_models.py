"""09 공용 — 모델 학습 함수 (그림 스크립트 c09_curves / c09_compare / c09_site 가 공유).

- prep: 학습 데이터로만 고분산 유전자 2,000개를 고르고 표준화 (누수 없음)
- train_mlp: 2000 → 256 → k, ReLU, dropout, AdamW, 검증 loss 기준 early stopping
- fit_l2 / fit_en / fit_rf / fit_mlp: 검증셋으로 하이퍼파라미터를 고른 뒤 전체 학습 데이터로 다시 학습
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, copy, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import balanced_accuracy_score as bacc
torch.set_num_threads(2)

def load():
    from c09_data import load_brca
    return load_brca()

def prep(Xtr, Xte, n_genes=2000):
    g = Xtr.var().nlargest(n_genes).index
    mu, sd = Xtr[g].mean(), Xtr[g].std() + 1e-8
    return ((Xtr[g]-mu)/sd).values.astype(np.float32), ((Xte[g]-mu)/sd).values.astype(np.float32)

def train_mlp(A, ya, V, yv, hidden=256, dropout=0.5, wd=1e-2, lr=1e-3, max_epochs=300,
              patience=30, batch=64, seed=0, history=False, early_stop=True):
    torch.manual_seed(seed)
    k = int(max(ya.max(), yv.max())) + 1
    net = nn.Sequential(nn.Linear(A.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, k))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lf = nn.CrossEntropyLoss()
    A_, ya_, V_, yv_ = map(torch.tensor, (A, ya, V, yv))
    best, best_state, best_ep, bad, hist = 1e9, None, 0, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(A_), generator=g)
        for i in range(0, len(A_), batch):
            idx = perm[i:i+batch]
            opt.zero_grad(); loss = lf(net(A_[idx]), ya_[idx]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            tl = lf(net(A_), ya_).item(); vl = lf(net(V_), yv_).item()
        hist.append((tl, vl))
        if vl < best - 1e-4:
            best, best_state, best_ep, bad = vl, copy.deepcopy(net.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience: break
    if early_stop:
        net.load_state_dict(best_state)
    net.eval()
    return (net, best_ep, hist) if history else (net, best_ep)

def predict(net, B):
    with torch.no_grad():
        return net(torch.tensor(B)).argmax(1).numpy()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def fit_l2(A, ya, V, yv, A_full, y_full):
    best = max([0.001, 0.01, 0.1, 1.0], key=lambda C: bacc(yv, LogisticRegression(C=C, max_iter=3000).fit(A, ya).predict(V)))
    return LogisticRegression(C=best, max_iter=3000).fit(A_full, y_full), best

def fit_en(A, ya, V, yv, A_full, y_full):
    mk = lambda C: LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=C, max_iter=1000, tol=1e-3, random_state=0)
    best = max([0.03, 0.1, 0.3, 1.0], key=lambda C: bacc(yv, mk(C).fit(A, ya).predict(V)))
    return mk(best).fit(A_full, y_full), best

def fit_rf(A, ya, V, yv, A_full, y_full):
    return RandomForestClassifier(500, n_jobs=2, random_state=0).fit(A_full, y_full), None

MLP_GRID = [(lr, wd, dp) for lr in (1e-3, 1e-4) for wd in (1e-4, 1e-2) for dp in (0.0, 0.5)]

def fit_mlp(A, ya, V, yv, A_full, y_full, seed=0):
    res = []
    for lr, wd, dp in MLP_GRID:
        net, ep = train_mlp(A, ya, V, yv, lr=lr, wd=wd, dropout=dp, seed=seed)
        res.append((bacc(yv, predict(net, V)), lr, wd, dp, ep))
    _, lr, wd, dp, ep = max(res, key=lambda r: r[0])
    # 고른 설정·epoch 수로 전체 학습 데이터에 다시 학습 (검증 없이 ep+1 epoch)
    net, _ = train_mlp(A_full, y_full, A_full[:8], y_full[:8], lr=lr, wd=wd, dropout=dp,
                       max_epochs=ep + 1, patience=10**9, seed=seed, early_stop=False)
    return FixedEpochs(net, ep + 1, lr, wd, dp), (lr, wd, dp, ep + 1)

class FixedEpochs:
    def __init__(self, net, ep, lr, wd, dp): self.net, self.ep = net, ep
    def predict(self, B): return predict(self.net, B)

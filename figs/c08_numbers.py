"""08 — 그림 없이 본문에만 쓰는 수치를 재현하는 스크립트.

§1-2 선형층 접기, §1-3 XOR 로지스틱 회귀, §2-3 MSE vs CE 기울기,
§3-1 학습률 0.1이 풀기 시작하는 시점, §3-2 한 번 갱신 후 loss,
§3-3 죽은 ReLU 개수, §4 파라미터 수.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

# ── §1-2 선형층 3개 = 행렬 하나 ───────────────────────────────────────
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(2, 16), nn.Linear(16, 16), nn.Linear(16, 1))
A, B, C = (m.weight for m in net)
W = C @ B @ A
b = C @ (B @ net[0].bias + net[1].bias) + net[2].bias
x = torch.randn(1000, 2)
print("§1-2 선형 3층 파라미터", sum(p.numel() for p in net.parameters()),
      "| 접은 행렬과 최대 차이", (net(x) - (x @ W.T + b)).abs().max().item())

# ── §1-3 XOR 로지스틱 회귀 ± 상호작용 항 ─────────────────────────────
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
y = np.array([0, 1, 1, 0])
Xi = np.c_[X, X[:, 0] * X[:, 1]]
for Cr in [1.0, 1e6]:
    m1 = LogisticRegression(C=Cr).fit(X, y)
    m2 = LogisticRegression(C=Cr).fit(Xi, y)
    print(f"§1-3 C={Cr:g}  x1,x2: acc {m1.score(X, y):.2f} coef {m1.coef_.round(4)}"
          f"  | +x1*x2: acc {m2.score(Xi, y):.2f}")

# ── §2-3 정답 0인데 틀리게 확신했을 때 z에 대한 기울기 ─────────────────
for z in [2, 4, 8]:
    p = 1 / (1 + np.exp(-z))
    mse, ce = 2 * p * p * (1 - p), p
    print(f"§2-3 z={z} p={p:.4f}  MSE {mse:.4g}  CE {ce:.4g}  배율 {ce / mse:.1f}")

# ── §3 XOR numpy (c08_training.py와 같은 구현) ────────────────────────
Y = y.reshape(-1, 1).astype(float)
sig = lambda z: 1 / (1 + np.exp(-z))


def train(H, act, seed, lr=1.0, epochs=5000):
    f = {"sigmoid": sig, "ReLU": lambda z: np.maximum(z, 0)}[act]
    df = {"sigmoid": lambda z, h: h * (1 - h),
          "ReLU": lambda z, h: (z > 0).astype(float)}[act]
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, H)); b1 = np.zeros(H)
    W2 = rng.normal(0, 1, (H, 1)); b2 = np.zeros(1)
    L = []
    for _ in range(epochs):
        z1 = X @ W1 + b1; h = f(z1); p = sig(h @ W2 + b2)
        L.append(-np.mean(Y * np.log(p + 1e-12) + (1 - Y) * np.log(1 - p + 1e-12)))
        dz2 = (p - Y) / 4; dW2 = h.T @ dz2; db2 = dz2.sum(0)
        dz1 = (dz2 @ W2.T) * df(z1, h); dW1 = X.T @ dz1; db1 = dz1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    solved = bool(((p > 0.5) == Y).all())
    dead = bool(((X @ W1 + b1) <= 0).all(axis=0).any())   # 네 입력 모두에서 꺼진 뉴런
    return np.array(L), solved, dead


L, _, _ = train(4, "sigmoid", 0, lr=0.1)
print("§3-1 lr=0.1: loss<0.6 처음 도달 epoch", int(np.argmax(L < 0.6)),
      "| loss<0.5", int(np.argmax(L < 0.5)))

x_, y_, w1, w2 = 2.0, 1.0, 0.5 - 0.01 * 24, 3.0 - 0.01 * 4
print("§3-2 한 번 갱신 후 w1", w1, "w2", w2, "L", round((w2 * max(w1 * x_, 0) - y_) ** 2, 4))

res = [train(2, "ReLU", s)[1:] for s in range(100)]
ok = [d for s, d in res if s]; bad = [d for s, d in res if not s]
print(f"§3-3 ReLU 은닉 2개: 성공 {len(ok)} (죽은 뉴런 {sum(ok)}) | "
      f"실패 {len(bad)} (죽은 뉴런 {sum(bad)})")

# ── §3-3 같은 값으로 초기화 (실습 2의 torch 코드와 같은 설정) ────────
Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
for c in [0.0, 0.5]:
    m = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid(), nn.Linear(4, 1))
    for q in m.parameters():
        nn.init.constant_(q, c)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    for _ in range(5001):
        opt.zero_grad(); loss = nn.BCEWithLogitsLoss()(m(Xt), Yt); loss.backward(); opt.step()
    acc = ((m(Xt) > 0).float() == Yt).float().mean().item()
    print(f"§3-3 상수 {c} 초기화: loss {loss.item():.4f} 정확도 {acc} "
          f"은닉 가중치 행 {m[0].weight.detach().numpy().round(3).tolist()}")

# ── §4-2 오믹스 규모 MLP 파라미터 수 ──────────────────────────────────
big = nn.Sequential(nn.Linear(20000, 256), nn.ReLU(), nn.Linear(256, 2))
print("§4-2 20000→256→2 파라미터", sum(p.numel() for p in big.parameters()))

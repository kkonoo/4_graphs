"""08 §3 — 학습률과 시드. XOR 4점, numpy 2층 NN (본문 🐍 실습과 같은 코드).

왼쪽: 학습률만 바꿨을 때 loss 곡선 (sigmoid 은닉 4개, 시드 0, 5000 epoch).
오른쪽: 은닉 뉴런 수 × 활성함수별로 시드 100개 중 XOR을 푼 횟수.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *

OUT = outdir("c08")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
y = np.array([[0], [1], [1], [0]], float)
sig = lambda z: 1 / (1 + np.exp(-z))
ACTS = {
    "sigmoid": (sig, lambda z, h: h * (1 - h)),
    "tanh": (np.tanh, lambda z, h: 1 - h ** 2),
    "ReLU": (lambda z: np.maximum(z, 0), lambda z, h: (z > 0).astype(float)),
}


def train(H=4, act="sigmoid", seed=0, lr=1.0, epochs=5000):
    f, df = ACTS[act]
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, H)); b1 = np.zeros(H)
    W2 = rng.normal(0, 1, (H, 1)); b2 = np.zeros(1)
    L = []
    for _ in range(epochs):
        z1 = X @ W1 + b1; h = f(z1); z2 = h @ W2 + b2; p = sig(z2)
        L.append(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
        dz2 = (p - y) / 4; dW2 = h.T @ dz2; db2 = dz2.sum(0)
        dz1 = (dz2 @ W2.T) * df(z1, h); dW1 = X.T @ dz1; db1 = dz1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    return np.array(L), bool(((p > 0.5) == y).all())


fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5),
                         gridspec_kw={"width_ratios": [1.15, 1]})

# ── 왼쪽: 학습률 ──────────────────────────────────────────────────────
ax = axes[0]
for lr, c in [(0.01, GREY), (0.1, OLIVE), (1, TEAL), (10, PURPLE), (30, RED)]:
    L, ok = train(lr=lr)
    ax.plot(np.arange(1, len(L) + 1), L, color=c, lw=2,
            label=f"η = {lr}  →  최종 {L[-1]:.4f}")
    print(f"lr={lr:<5} loss[0]={L[0]:.3f} loss[1000]={L[1000]:.4f} "
          f"final={L[-1]:.4f} max={L.max():.2f} solved={ok}")
ax.axhline(np.log(2), color=NAVY, ls=":", lw=1.2)
ax.text(4800, np.log(2) * 1.25, "ln 2 = 0.693 (항상 0.5라고 답할 때)",
        color=NAVY, fontsize=9.5, ha="right")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_ylabel("loss (binary cross-entropy)")
ax.set_title("학습률 η만 바꿨을 때", color=NAVY)
ax.legend(frameon=False, fontsize=9.5, loc="lower left")

# ── 오른쪽: 시드 100개 중 성공 횟수 ──────────────────────────────────
ax = axes[1]
Hs = [2, 3, 4, 8, 16]
w = 0.26
for k, (act, c) in enumerate([("sigmoid", ORANGE), ("tanh", PURPLE), ("ReLU", TEAL)]):
    wins = [sum(train(H=H, act=act, seed=s)[1] for s in range(100)) for H in Hs]
    print(act, dict(zip(Hs, wins)))
    xs = np.arange(len(Hs)) + (k - 1) * w
    ax.bar(xs, wins, width=w, color=c, label=act)
    for x_, v in zip(xs, wins):
        ax.text(x_, v + 1.5, str(v), ha="center", fontsize=8.5, color="#333")
ax.set_xticks(np.arange(len(Hs))); ax.set_xticklabels(Hs)
ax.set_xlabel("은닉 뉴런 수")
ax.set_ylabel("XOR을 푼 횟수 (시드 100개 중)")
ax.set_ylim(0, 124)
ax.set_title("같은 코드, 시드만 다르게", color=NAVY)
ax.legend(frameon=False, loc="upper center", ncol=3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "training.png"))
print("wrote", os.path.join(OUT, "training.png"))

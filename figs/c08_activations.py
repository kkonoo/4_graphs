"""08 §2 — 활성함수와 기울기 소실.

왼쪽: 활성함수 3종. 가운데: 그 도함수 (sigmoid 최대 0.25).
오른쪽: 은닉 10층 MLP에서 층별 가중치 기울기 크기 (실측, 시드 20개 평균).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import torch
import torch.nn as nn

OUT = outdir("c08")

z = np.linspace(-5, 5, 400)
sig = 1 / (1 + np.exp(-z))
acts = {"sigmoid": (sig, sig * (1 - sig), ORANGE),
        "tanh": (np.tanh(z), 1 - np.tanh(z) ** 2, PURPLE),
        "ReLU": (np.maximum(z, 0), (z > 0).astype(float), TEAL)}

# ── 층별 기울기 실측 ─────────────────────────────────────────────────
DEPTH, WIDTH, BATCH, SEEDS = 10, 64, 256, 20


def layer_grads(act, seed, he=False):
    torch.manual_seed(seed)
    layers = [nn.Linear(20, WIDTH), act()]
    for _ in range(DEPTH - 1):
        layers += [nn.Linear(WIDTH, WIDTH), act()]
    layers += [nn.Linear(WIDTH, 1)]
    net = nn.Sequential(*layers)
    if he:                                          # He(Kaiming) 초기화
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    X = torch.randn(BATCH, 20)
    y = torch.randint(0, 2, (BATCH,)).float()
    loss = nn.BCEWithLogitsLoss()(net(X).squeeze(1), y)
    loss.backward()
    lin = [m for m in net if isinstance(m, nn.Linear)]
    return np.array([m.weight.grad.norm().item() for m in lin])


res = {}
for name, act, he in [("sigmoid", nn.Sigmoid, False), ("tanh", nn.Tanh, False),
                      ("ReLU", nn.ReLU, False), ("ReLU + He 초기화", nn.ReLU, True)]:
    G = np.array([layer_grads(act, s, he) for s in range(SEEDS)])
    res[name] = np.exp(np.log(G).mean(0))          # 기하평균
    r = res[name][0] / res[name][-2]
    print(f"{name:14s} 1층 {res[name][0]:.2e}  10층 {res[name][-2]:.2e}  "
          f"출력층 {res[name][-1]:.2e}  1층/10층 = {r:.2e}")

print("0.25^10 =", 0.25 ** 10)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3),
                         gridspec_kw={"width_ratios": [1, 1, 1.25]})
for name, (f, d, c) in acts.items():
    axes[0].plot(z, f, color=c, lw=2.2, label=name)
    axes[1].plot(z, d, color=c, lw=2.2, label=name)
axes[0].set_ylim(-1.2, 3); axes[0].axhline(0, color=LGREY, lw=0.8, zorder=0)
axes[0].set_title("활성함수", color=NAVY); axes[0].set_xlabel("z")
axes[0].legend(frameon=False, loc="upper left")
axes[1].axhline(0.25, color=ORANGE, ls="--", lw=1)
axes[1].text(-4.9, 0.29, "sigmoid 최대 0.25", color=ORANGE, fontsize=10)
axes[1].set_title("도함수 (기울기가 통과하는 비율)", color=NAVY)
axes[1].set_xlabel("z"); axes[1].set_ylim(-0.05, 1.1)

ax = axes[2]
L = np.arange(1, DEPTH + 2)
for name, c, ls in [("sigmoid", ORANGE, "-"), ("tanh", PURPLE, "-"),
                    ("ReLU", TEAL, "-"), ("ReLU + He 초기화", NAVY, "--")]:
    ax.plot(L, res[name], "o", ls=ls, color=c, lw=2, ms=5, label=name)
ax.set_yscale("log")
ax.set_xticks(L); ax.set_xticklabels([str(i) for i in L[:-1]] + ["출력"])
ax.set_xlabel("층 (입력 쪽 → 출력 쪽)")
ax.set_ylabel("가중치 기울기 크기  ‖∂L/∂W‖")
ax.set_title(f"은닉 {DEPTH}층 MLP의 층별 기울기 (실측)", color=NAVY)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "activations.png"))
print("wrote", os.path.join(OUT, "activations.png"))

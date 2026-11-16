"""08 §2 — 활성함수와 기울기 소실.

왼쪽 위: 활성함수 3종. 왼쪽 아래: 그 도함수 (sigmoid 최대 0.25).
오른쪽: 은닉 10층 MLP에서 층별 가중치 기울기 크기 (실측, 시드 20개 기하평균).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import torch
import torch.nn as nn
from matplotlib.ticker import FuncFormatter

z = np.linspace(-5, 5, 400)
sig = 1 / (1 + np.exp(-z))
C_SIG, C_TANH, C_RELU = CAT[0], CAT[1], CAT[2]      # 나란한 세 활성함수
acts = {"sigmoid": (sig, sig * (1 - sig), C_SIG),
        "tanh": (np.tanh(z), 1 - np.tanh(z) ** 2, C_TANH),
        "ReLU": (np.maximum(z, 0), (z > 0).astype(float), C_RELU)}

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

# ── 그림 ────────────────────────────────────────────────────────────
plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
fig = plt.figure(figsize=(8.4, 5.3))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.45], hspace=0.62, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

minus = FuncFormatter(lambda v, p: f"{v:g}".replace("-", "\u2212"))

def marker(ax, text, y=1.06):
    ax.text(0, y, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT)

# ① 활성함수
for name, (f, d, c) in acts.items():
    ax1.plot(z, f, color=c, lw=2.2, zorder=3)
    ax2.plot(z, d, color=c, lw=2.2, zorder=3)
ax1.axhline(0, color=LINE, lw=0.9, zorder=0)
ax1.set_ylim(-1.2, 3); ax1.set_yticks([-1, 0, 1, 2, 3])
ax1.set_xticks([-4, -2, 0, 2, 4])
ax1.xaxis.set_major_formatter(minus); ax1.yaxis.set_major_formatter(minus)
ax1.set_xlabel("z", fontsize=11.5, labelpad=2)
ax1.text(2.0, 2.5, "ReLU", color=C_RELU, fontsize=11.5, fontweight="bold", ha="right", va="center")
ax1.text(-4.8, -0.8, "tanh", color=C_TANH, fontsize=11.5, fontweight="bold", ha="left", va="bottom")
ax1.text(-1.2, 0.62, "sigmoid", color=C_SIG, fontsize=11.5, fontweight="bold", ha="right", va="center")
marker(ax1, "① 활성함수")

# ② 도함수
ax2.axhline(0.25, color=ACCENT_INK, ls=(0, (4, 2.5)), lw=1.3, zorder=2)
ax2.text(4.95, 0.31, "sigmoid\n최대 0.25", color=ACCENT_INK, fontsize=11, fontweight="bold",
         ha="right", va="bottom", linespacing=1.15)
ax2.set_ylim(-0.05, 1.1); ax2.set_yticks([0, 0.5, 1.0])
ax2.set_xticks([-4, -2, 0, 2, 4])
ax2.xaxis.set_major_formatter(minus)
ax2.set_xlabel("z", fontsize=11.5, labelpad=2)
marker(ax2, "② 도함수 (기울기 통과 비율)")

# ③ 층별 기울기
L = np.arange(1, DEPTH + 2)
ax3.grid(axis="y", color=LINE, lw=0.7, zorder=0)
for name, c, ls in [("sigmoid", C_SIG, "-"), ("tanh", C_TANH, "-"),
                    ("ReLU", C_RELU, "-"), ("ReLU + He 초기화", C_RELU, (0, (4, 2.5)))]:
    ax3.plot(L, res[name], "o", ls=ls, color=c, lw=2, ms=4.5, zorder=3)
ax3.set_yscale("log")
ax3.set_ylim(1.5e-10, 6)
ax3.set_xlim(0.5, 11.5)
ax3.set_xticks(L); ax3.set_xticklabels([str(i) for i in L[:-1]] + ["출력"])
ax3.set_xlabel("층 (입력 쪽 → 출력 쪽)", fontsize=11.5)
ax3.set_ylabel("가중치 기울기 크기  ‖∂L/∂W‖", fontsize=11.5)
kw = dict(fontsize=11.5, fontweight="bold", ha="left")
ax3.text(1.0, 0.05, "ReLU + He 초기화", color=C_RELU, va="top", **kw)
ax3.text(1.0, 1.6e-3, "tanh", color=C_TANH, va="bottom", **kw)
ax3.text(1.0, 7e-6, "ReLU", color=C_RELU, va="top", **kw)
ax3.text(2.4, 1.5e-9, "sigmoid", color=C_SIG, va="top", **kw)
# taller axes: smaller relative offset keeps the marker level with ①
marker(ax3, f"③ 은닉 {DEPTH}층 MLP의 층별 기울기", y=1.028)

save(fig, img("c08", "activations.png"))

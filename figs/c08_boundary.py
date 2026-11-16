"""08 §4 — 층을 쌓으면 결정경계가 휜다.

노이즈가 있는 XOR (마커 A, B 중 하나만 높으면 1). 같은 데이터, 같은 학습 설정,
모델만 바꿔서 결정경계를 그린다 (2 × 3: 첫 칸은 데이터만).
패널 위에 train/test 정확도를 적는다. 학습 200점, 테스트 2,000점, Adam lr=0.01, 3,000 epoch.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap


def make_xor(n, rng):
    centers = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], float)
    lab = np.array([0, 0, 1, 1])
    k = rng.integers(0, 4, n)
    X = centers[k] + rng.normal(0, 0.55, (n, 2))
    return X.astype(np.float32), lab[k].astype(np.float32)


rng = np.random.default_rng(8)
Xtr, ytr = make_xor(200, rng)
Xte, yte = make_xor(2000, rng)


def fit(model, X, y, epochs=3000, lr=0.01):
    X, y = torch.tensor(X), torch.tensor(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(X).squeeze(1), y)
        loss.backward()
        opt.step()
    return model


def acc(model, X, y):
    with torch.no_grad():
        p = model(torch.tensor(X)).squeeze(1) > 0
    return (p.numpy() == (y > 0.5)).mean()


specs = [
    ("로지스틱 회귀 (뉴런 1개)", "② 로지스틱 회귀", lambda: nn.Linear(2, 1)),
    ("선형 3층 (2→16→16→1), 활성함수 없음", "③ 선형 3층, 활성함수 없음",
     lambda: nn.Sequential(nn.Linear(2, 16), nn.Linear(16, 16), nn.Linear(16, 1))),
    ("ReLU 은닉 1층 × 2뉴런", "④ ReLU 은닉 1층 × 2",
     lambda: nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))),
    ("ReLU 은닉 1층 × 8뉴런", "⑤ ReLU 은닉 1층 × 8",
     lambda: nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))),
    ("ReLU 은닉 2층 × 64뉴런", "⑥ ReLU 은닉 2층 × 64",
     lambda: nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64),
                           nn.ReLU(), nn.Linear(64, 1))),
]

g = np.linspace(-3, 3, 300).astype(np.float32)
xx, yy = np.meshgrid(g, g)
grid = np.c_[xx.ravel(), yy.ravel()]

# ── 색: 두 클래스 = 나란한 두 범주 ───────────────────────────────────────
C0, C1 = CAT[2], CAT[0]                 # 0 = 청록 (둘 다 높음/낮음), 1 = 보라 (하나만 높음)
cmap = ListedColormap([mix(C0, 0.14), mix(C1, 0.13)])

# ── 배치 (인치): 2 × 3, 정사각 패널 ───────────────────────────────────────
W, H = 8.4, 6.6
S, GX, LM = 2.45, 0.3, 0.4
rows_y = [3.6, 0.35]
fig = plt.figure(figsize=(W, H))
axes = []
for r in range(2):
    for c in range(3):
        x0 = LM + c * (S + GX)
        axes.append(fig.add_axes([x0 / W, rows_y[r] / H, S / W, S / H]))


def frame(ax, i):
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(LINE); sp.set_linewidth(0.8)
    if i >= 3:
        ax.set_xlabel("마커 A", fontsize=11.5, labelpad=4)
    if i % 3 == 0:
        ax.set_ylabel("마커 B", fontsize=11.5, labelpad=4)


def scatter(ax):
    ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=11, color=C0, lw=0, zorder=3)
    ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=11, color=C1, lw=0, zorder=3)


def header(ax, marker, sub):
    ax.text(0.5, 1.11, marker, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT)
    ax.text(0.5, 1.025, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, color=TEXT2)


# 첫 패널: 데이터만
scatter(axes[0]); frame(axes[0], 0)
header(axes[0], "① 학습 데이터", f"{len(Xtr)}점 · 보라 = 둘 중 하나만 높음")

for i, (ax, (name, marker, build)) in enumerate(zip(axes[1:], specs), start=1):
    torch.manual_seed(0)
    m = fit(build(), Xtr, ytr)
    a_tr, a_te = acc(m, Xtr, ytr), acc(m, Xte, yte)
    with torch.no_grad():
        z = m(torch.tensor(grid)).squeeze(1).numpy().reshape(xx.shape)
    ax.contourf(xx, yy, z > 0, levels=[-0.5, 0.5, 1.5], cmap=cmap, zorder=1)
    ax.contour(xx, yy, z, levels=[0], colors=TEXT, linewidths=1.3, zorder=2)
    scatter(ax); frame(ax, i)
    header(ax, marker, f"train {a_tr:.2f} · test {a_te:.2f}")
    print(f"{name.splitlines()[0]:28s} train={a_tr:.3f} test={a_te:.3f}")
    if "2뉴런" in name:      # 본문 §4-1: 뉴런 하나가 죽었는가
        with torch.no_grad():
            act = (torch.relu(m[0](torch.tensor(Xtr))) > 0).float().mean(0)
        print("   은닉 뉴런별 활성 비율 (학습 200점):", act.numpy().round(3))

save(fig, img("c08", "boundary.png"))

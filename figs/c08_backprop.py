"""08 §3-2 — 역전파 = 연쇄법칙, 손으로 한 번.

x → z = w1·x → h = ReLU(z) → ŷ = w2·h → L = (ŷ − y)²
위 줄: 순전파 값(왼쪽 → 오른쪽). 아래 줄: 역전파 기울기(오른쪽 → 왼쪽), 화살표마다 곱하는 국소 기울기.
숫자는 본문 표와 같은 손계산 (x = 2, y = 1, w1 = 0.5, w2 = 3).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *

# ── 손계산 (본문 §3-2) ─────────────────────────────────────────────────────
x, y, w1, w2 = 2.0, 1.0, 0.5, 3.0
z = w1 * x
h = max(0.0, z)
yhat = w2 * h
L = (yhat - y) ** 2
relu_d = 1.0 if z > 0 else 0.0
dyhat = 2 * (yhat - y)
dw2 = dyhat * h
dh = dyhat * w2
dz = dh * relu_d
dw1 = dz * x
print(f"forward : z = {z:g}, h = {h:g}, yhat = {yhat:g}, L = {L:g}")
print(f"backward: dL/dyhat = {dyhat:g}, dL/dh = {dh:g}, dL/dz = {dz:g}")
print(f"weights : dL/dw1 = {dw1:g}, dL/dw2 = {dw2:g}")

g = lambda v: f"{v:g}"
FW, BW = CAT[0], CAT[2]                 # 순전파 · 역전파 (나란한 두 흐름)
BWT = shade(BW, 0.12)                   # 흰 바탕 위 역전파 글자

fig, ax = new(8.4, 4.0)

xs = [0.75, 2.45, 4.15, 5.85, 7.55]     # x, z, h, ŷ, L 칸 중심
bw, yc, bh = 1.36, 2.45, 0.8            # 칸 폭, 중심 높이, 높이

# ── 순전파 줄 ───────────────────────────────────────────────────────────────
nodes = [("입력  $x$", x), ("$z = w_1 x$", z), (r"$h = \mathrm{ReLU}(z)$", h),
         (r"$\hat{y} = w_2 h$", yhat), (r"$L = (\hat{y} - y)^2$", L)]
for cx, (expr, val) in zip(xs, nodes):
    box(ax, cx - bw / 2, yc - bh / 2, bw, bh, fc=mix(FW, 0.10), ec="none")
    label(ax, cx, yc + 0.17, expr, fs=11.5, c=TEXT)
    label(ax, cx, yc - 0.19, f"= {g(val)}", fs=13, w="bold", c=FW)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, a + bw / 2 + 0.03, yc, b - bw / 2 - 0.03, yc, color=FW, ms=11)

# 매개변수와 정답 (위에서 들어오는 값)
yp = 3.55
for cx, text in [(xs[1], f"$w_1$ = {g(w1)}"), (xs[3], f"$w_2$ = {g(w2)}"), (xs[4], f"$y$ = {g(y)}")]:
    pill(ax, cx, yp, text, fc=mix(FW, 0.12), tc=FW, fs=11.5, h=0.36)
    arrow(ax, cx, yp - 0.2, cx, yc + bh / 2 + 0.04, color=mix(FW, 0.6), ms=10, lw=1.4)
label(ax, xs[0], yp, "순전파  →", fs=12.5, w="bold", c=FW)

# ── 역전파 줄 ───────────────────────────────────────────────────────────────
yg = 1.25
grads = {3: (r"$\partial L / \partial \hat{y}$", dyhat),
         2: (r"$\partial L / \partial h$", dh),
         1: (r"$\partial L / \partial z$", dz)}
wid = {}
for i, (sym, val) in grads.items():
    wid[i] = pill(ax, xs[i], yg, f"{sym} = {g(val)}", fc=mix(BW, 0.14), tc=BWT, fs=11.5, h=0.38)

# L → ∂L/∂ŷ
arrow(ax, xs[4] - 0.25, yc - bh / 2 - 0.04, xs[3] + wid[3] / 2 + 0.06, yg + 0.05, color=BW, ms=11)
label(ax, 7.08, 1.36, r"$2(\hat{y} - y)$" + f" = 2 × {g(yhat - y)}", fs=11, c=BWT, ha="left")

# ∂L/∂ŷ → ∂L/∂h → ∂L/∂z, 국소 기울기를 곱함
steps = [(3, 2, f"× $w_2$ = {g(w2)}"), (2, 1, r"× $\mathrm{ReLU}'(z)$" + f" = {g(relu_d)}")]
for a, b, text in steps:
    arrow(ax, xs[a] - wid[a] / 2 - 0.06, yg, xs[b] + wid[b] / 2 + 0.06, yg, color=BW, ms=11)
    label(ax, (xs[a] + xs[b]) / 2, yg + 0.37, text, fs=11, c=BWT)
label(ax, xs[0], yg, "←  역전파", fs=12.5, w="bold", c=BW)

# 가중치의 기울기 (맨 아래)
yw = 0.22
for i, sym, loc, val in [(3, "$w_2$", f"× $h$ = {g(h)}", dw2), (1, "$w_1$", f"× $x$ = {g(x)}", dw1)]:
    arrow(ax, xs[i], yg - 0.21, xs[i], yw + 0.23, color=BW, ms=11)
    label(ax, xs[i] + 0.12, (yg + yw) / 2 + 0.02, loc, fs=11, c=BWT, ha="left")
    pill(ax, xs[i], yw, r"$\partial L / \partial$" + sym + f" = {g(val)}", fc=BW, fs=12, h=0.4)

save(fig, img("c08", "backprop.png"))

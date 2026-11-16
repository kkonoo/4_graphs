"""08 §1-1 — 로지스틱 회귀는 뉴런 하나, MLP는 그 사이에 f를 넣은 것.

위: 입력 → 가중치 → Σ (z = w·x + b) → σ → 0~1.
아래: 입력 → 은닉층 (W₁x + b₁) → f → Σ (W₂·… + b₂) → σ → 출력.
두 줄의 끝(Σ → σ → 출력)을 같은 자리에 두어, 새로 생긴 것이 f 하나임을 보인다.
설명용 도식 (데이터 없음).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *

IN, SUM, OUT = STEPS[2], CAT[1], PRIMARY        # 입력 · 가중합 · 출력 (두 줄에서 같은 뜻 = 같은 색)
EDGE = "#C3C8CF"

fig, ax = new(8.3, 5.7)

def node(x, y, fc, text, r=0.2, fs=11.5, ec="none", tc=None, lw=0):
    ax.add_patch(mp.Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=4))
    label(ax, x, y, text, fs=fs, w="bold", c=tc or on(fc), z=5)

def edge(p, q, c=EDGE, lw=1.1, ls="-"):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, ls=ls, zorder=2, solid_capstyle="round")

def sigma_box(x, y):
    box(ax, x - 0.28, y - 0.28, 0.56, 0.56, fc=OUT, ec="none", r=0.1, z=4)
    label(ax, x, y, "σ", fs=15, w="bold", c=on(OUT))

XS, XSIG, XOUT = 3.3, 4.35, 5.35               # Σ · σ · 출력: 두 줄에서 같은 x
XF = 5.95                                       # 오른쪽 식

# ── ① 로지스틱 회귀 = 뉴런 하나 ───────────────────────────────────────────
y1 = 4.5
label(ax, 0.05, 5.55, "① 로지스틱 회귀 = 뉴런 하나", fs=12.5, w="bold", ha="left")
ins = [(0.45, 4.95), (0.45, 4.3)]
edge((0.65, ins[0][1]), (XS - 0.3, y1 + 0.08))
edge((0.65, ins[1][1]), (XS - 0.3, y1 - 0.02))
edge((0.6, 3.85), (XS - 0.29, y1 - 0.12), ls=(0, (3, 2)))
for (x, y), t in zip(ins, ["x₁", "x₂"]):
    node(x, y, IN, t)
label(ax, 0.45, 3.85, "⋮", fs=13, c=TEXT2)
label(ax, 1.85, 4.9, "w₁", fs=12, w="bold", c=TEXT2)
label(ax, 1.85, 4.55, "w₂", fs=12, w="bold", c=TEXT2)

node(XS, y1, SUM, "Σ", r=0.3, fs=15)
label(ax, XS, 5.22, "b", fs=12, w="bold", c=TEXT2)
arrow(ax, XS, 5.08, XS, y1 + 0.33, ms=10, lw=1.3)
label(ax, XS, 4.1, "가중합\nz = w·x + b", fs=11, c=TEXT, va="top", linespacing=1.35)

arrow(ax, XS + 0.33, y1, XSIG - 0.32, y1)
sigma_box(XSIG, y1)
label(ax, XSIG, 4.1, "sigmoid", fs=11, c=TEXT, va="top")
arrow(ax, XSIG + 0.31, y1, XOUT - 0.26, y1)
node(XOUT, y1, "white", "ŷ", r=0.22, ec=OUT, lw=1.6, tc=OUT)
label(ax, XOUT, 4.1, "0~1 사이 값", fs=11, c=TEXT, va="top")

label(ax, XF, y1 + 0.13, "σ(w·x + b)", fs=13, w="bold", ha="left")
label(ax, XF, y1 - 0.25, "가중합 → sigmoid", fs=11, c=TEXT2, ha="left")

# ── ② 다층 신경망 (MLP) ───────────────────────────────────────────────────
y2 = 1.6
label(ax, 0.05, 3.3, "② 다층 신경망 (MLP)", fs=12.5, w="bold", ha="left")
ins2 = [(0.45, 2.1), (0.45, 1.5)]
XH = 1.65
hid = [(XH, y) for y in [2.35, 1.85, 1.35, 0.85]]
for p in ins2:
    for h in hid:
        edge((p[0] + 0.2, p[1]), (h[0] - 0.19, h[1]), lw=0.9)
for h in hid:
    edge((0.6, 1.0), (h[0] - 0.19, h[1]), lw=0.8, ls=(0, (3, 2)))
for (x, y), t in zip(ins2, ["x₁", "x₂"]):
    node(x, y, IN, t)
label(ax, 0.45, 1.0, "⋮", fs=13, c=TEXT2)
label(ax, 1.0, 2.5, "W₁", fs=12, w="bold", c=TEXT2)

# 활성함수 f: 은닉층 전체에 한 번 — 그림에서 유일한 강조색
FX0, FX1 = 2.15, 2.5
for h in hid:
    edge((h[0] + 0.19, h[1]), (FX0, h[1]), lw=1.1)
    edge((FX1, h[1]), (XS - 0.29, y2 + (h[1] - y2) * 0.25), lw=0.9)
for h in hid:
    node(*h, SUM, "Σ", r=0.19, fs=11)
box(ax, FX0, 0.62, FX1 - FX0, 1.96, fc=ACCENT, ec="none", r=0.1, z=4)
label(ax, (FX0 + FX1) / 2, 1.6, "f", fs=16, w="bold", c=ON_ACCENT)
label(ax, (FX0 + FX1) / 2, 2.8, "새로 생긴 것: 활성함수 f", fs=11.5, w="bold", c=TEXT)
label(ax, 2.95, 2.25, "W₂", fs=12, w="bold", c=TEXT2)
label(ax, XH, 0.45, "W₁x + b₁", fs=11, c=TEXT, va="top")

node(XS, y2, SUM, "Σ", r=0.3, fs=15)
label(ax, XS, 1.18, "W₂·(…) + b₂", fs=11, c=TEXT, va="top")
arrow(ax, XS + 0.33, y2, XSIG - 0.32, y2)
sigma_box(XSIG, y2)
label(ax, XSIG, 1.18, "sigmoid", fs=11, c=TEXT, va="top")
arrow(ax, XSIG + 0.31, y2, XOUT - 0.26, y2)
node(XOUT, y2, "white", "ŷ", r=0.22, ec=OUT, lw=1.6, tc=OUT)
label(ax, XOUT, 1.18, "0~1", fs=11, c=TEXT, va="top")

label(ax, XF, y2 + 0.13, "σ(W₂ · f(W₁x + b₁) + b₂)", fs=13, w="bold", ha="left")
label(ax, XF, y2 - 0.38, "f가 없으면 행렬 하나로\n접혀 다시 선형 (§1-2)", fs=11, c=TEXT2, ha="left",
      linespacing=1.35)

save(fig, img("c08", "neuron.png"))

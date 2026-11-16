"""09 §1-1, §1-2 — train · validation · test의 역할과 누수가 생기는 두 곳.

설명용 도식 (데이터 없음). 위: 세 부분의 역할과 보는 횟수. 아래: 괄호 = 계산하거나 고를 때 쓰는 부분.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *

TR, VA, TE = CAT[0], CAT[1], CAT[2]          # train · validation · test (나란한 세 역할)
COLS = [TR, VA, TE]

fig, ax = new(8.4, 5.2)

def segs(x0, w, gap):
    u = w - 2 * gap
    ws = [u * 0.5, u * 0.25, u * 0.25]
    xs = [x0, x0 + ws[0] + gap, x0 + ws[0] + ws[1] + 2 * gap]
    return list(zip(xs, ws))

def bar(x0, w, y, h, texts, fs, gap=0.05, bold=True):
    S = segs(x0, w, gap)
    for (x, ww), c, t in zip(S, COLS, texts):
        box(ax, x, y, ww, h, fc=c, ec="none", r=0.06)
        label(ax, x + ww / 2, y + h / 2, t, fs=fs, w="bold" if bold else "regular", c=on(c))
    return S

def bracket(x0, x1, y, c=TEXT):
    ax.plot([x0, x0, x1, x1], [y - 0.08, y, y, y - 0.08], color=c, lw=1.5,
            solid_joinstyle="miter", solid_capstyle="butt", zorder=3)

# ── ① 세 가지 역할 ─────────────────────────────────────────────────────────
label(ax, 0.05, 5.05, "① 세 가지 역할", fs=12.5, w="bold", ha="left")
S = bar(0.05, 8.3, 4.25, 0.5, ["train · 학습", "validation · 고름", "test · 보고"], 12)
descs = ["가중치를 학습", "모델·하이퍼파라미터·\n멈출 시점을 고름", "최종 성능을 보고"]
for (x, w), d in zip(S, descs):
    label(ax, x + w / 2, 4.1, d, fs=11, c=TEXT2, va="top", linespacing=1.35)
pills = [("매 epoch 봄", mix(TR, 0.15), TEXT), ("고를 때마다 봄", mix(VA, 0.15), TEXT),
         ("한 번만 봄", ACCENT, ON_ACCENT)]          # 강조색은 여기 한 곳
for (x, w), (t, fc, tc) in zip(S, pills):
    pill(ax, x + w / 2, 3.36, t, fc=fc, tc=tc, fs=11, h=0.32)

# ── ② 누수가 생기는 곳 ─────────────────────────────────────────────────────
label(ax, 0.05, 2.8, "② 누수가 생기는 곳", fs=12.5, w="bold", ha="left")
cols = [
    (0.05, "특징 선택 · 표준화",
     [("나누기 전에 전체 데이터로 계산", (0, 3)), ("train으로만 계산해 그대로 적용", (0, 1))],
     "누수인 이유: 전체로 고르면 test의 우연한 상관까지 고릅니다"),
    (4.4, "모델 · 하이퍼파라미터 고르기",
     [("test 점수로 고름", (2, 3)), ("validation 또는 nested CV로 고름", (1, 2))],
     "누수인 이유: test로 고르는 순간 test는 validation이 됩니다"),
]
CW = 3.95
rows = [(2.1, 1.83, 1.42), (1.15, 0.88, 0.47)]     # (pill y, bracket y, bar bottom)
tags = [("누수", TEXT, "white"), ("올바름", LINE, TEXT)]
for x0, head, items, foot in cols:
    label(ax, x0, 2.45, head, fs=11.5, w="bold", ha="left")
    for (txt, (i0, i1)), (py, by, bb), (tag, fc, tc) in zip(items, rows, tags):
        pw = pill(ax, x0, py, tag, fc=fc, tc=tc, fs=11, h=0.3, ha="left")
        label(ax, x0 + pw + 0.1, py, txt, fs=11, ha="left")
        Sm = bar(x0, CW, bb, 0.28, ["train", "validation", "test"], 11, gap=0.04, bold=False)
        bracket(Sm[i0][0], Sm[i1 - 1][0] + Sm[i1 - 1][1], by)
    label(ax, x0, 0.3, foot, fs=11, c=TEXT2, ha="left", va="top")

print("설명용 도식: train 학습(매 epoch) · validation 고름(고를 때마다) · test 보고(한 번)")
print("누수 두 곳: 나누기 전 전체 데이터로 특징 선택·표준화 / test 점수로 모델 고르기")
save(fig, img("c09", "split.png"))

"""10 §1-2 — 1D 합성곱 = 학습되는 PWM 스캔 (설명용).

필터(4 염기 × 폭) 하나가 one-hot 서열 위를 한 칸씩 밀리며 자리마다 점수를 내고,
모든 위치에 같은 필터(가중치 공유), 전역 max pooling은 가장 센 반응 하나만 남긴다.
그림을 위해 필터 폭을 모티프 길이 7로 줄였다 (실제 모델은 폭 12).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *

SEQ = "GCATAGTTGACTCAAGCT"          # 설명용 서열, 7번 자리(0부터)에 TGACTCA
MOTIF = "TGACTCA"
B = "ACGT"
L, K = len(SEQ), len(MOTIF)
X = np.zeros((4, L)); X[[B.index(c) for c in SEQ], np.arange(L)] = 1
W = np.full((4, K), 0.1)
W[[B.index(c) for c in MOTIF], np.arange(K)] = 1.0
W[B.index("C"), 0] = 0.4; W[B.index("G"), 4] = 0.3     # PWM처럼 약한 가중치 몇 개
scores = np.array([(W * X[:, i:i + K]).sum() for i in range(L - K + 1)])
best = int(scores.argmax())
print("서열", SEQ, " 길이", L, " 필터 폭", K)
print("점수", " ".join(f"{v:.1f}" for v in scores))
print(f"최대 점수 {scores.max():.1f} (위치 {best}, 창 {SEQ[best:best + K]})")

NEUTRAL = "#9AA1AB"
ONE = mix(CAT[2], 0.8)        # one-hot의 1
s, x0 = 0.34, 0.75            # 칸 크기, 서열 왼쪽 끝 (인치)
def cx(j): return x0 + (j + 0.5) * s

fig, ax = new(8.4, 5.3)

def cell(x, y, fc, ec="white", lw=1.2, z=2):
    ax.add_patch(mp.Rectangle((x, y), s, s, fc=fc, ec=ec, lw=lw, zorder=z))

# ── 필터 (학습되는 PWM) ─────────────────────────────────────────────
ft = 4.95
fx0 = x0 + best * s
for r in range(4):
    label(ax, fx0 - 0.13, ft - (r + 0.5) * s, B[r], fs=11, c=TEXT2)
    for j in range(K):
        cell(fx0 + j * s, ft - (r + 1) * s, mix(PRIMARY, 0.9 * W[r, j]))
for j in range(K):
    label(ax, fx0 + (j + 0.5) * s, ft + 0.17, MOTIF[j], fs=11, w="bold", c=PRIMARY)
ax.add_patch(mp.Rectangle((fx0, ft - 4 * s), K * s, 4 * s, fc="none", ec=PRIMARY, lw=1.6, zorder=4))
label(ax, 5.75, ft - 0.25, "필터 = 학습되는 PWM", fs=12, w="bold", ha="left")
label(ax, 5.75, ft - 0.6, "4 염기 × 폭 7", fs=11, c=TEXT2, ha="left")
label(ax, 5.75, ft - 0.9, "설명용 · 실제 모델은 폭 12", fs=11, c=TEXT2, ha="left")
fc_x = fx0 + K * s / 2
arrow(ax, fc_x, ft - 4 * s - 0.05, fc_x, 3.27, color=PRIMARY, lw=1.4)

# ── one-hot 서열 ────────────────────────────────────────────────────
st = 3.2
for r in range(4):
    label(ax, x0 - 0.13, st - (r + 0.5) * s, B[r], fs=11, c=TEXT2)
    for j in range(L):
        cell(x0 + j * s, st - (r + 1) * s, ONE if X[r, j] else "white", ec=LINE, lw=0.8)
for j in range(L):
    label(ax, cx(j), st - 4 * s - 0.17, SEQ[j], fs=11, w="bold")
ax.add_patch(mp.Rectangle((x0, st - 4 * s), K * s, 4 * s, fc="none", ec=NEUTRAL, lw=1.4,
                          ls=(0, (4, 3)), zorder=4))
label(ax, x0 + K * s / 2, st + 0.18, "같은 필터를 위치마다 (가중치 공유)", fs=11, c=TEXT2)
ax.add_patch(mp.Rectangle((fx0, st - 4 * s), K * s, 4 * s, fc="none", ec=PRIMARY, lw=2.0, zorder=5))
label(ax, 7.05, 2.78, "겹친 칸의 가중치를", fs=11, c=TEXT2, ha="left")
label(ax, 7.05, 2.52, "모두 더한 값이", fs=11, c=TEXT2, ha="left")
label(ax, 7.05, 2.26, "그 자리의 점수", fs=11, c=TEXT2, ha="left")

# ── 점수 막대 ──────────────────────────────────────────────────────
yb, H = 0.3, 1.0
ax.plot([x0, x0 + L * s], [yb, yb], color=NEUTRAL, lw=1.0, zorder=2)
label(ax, x0 - 0.05, yb + 0.45, "점수", fs=11, c=TEXT2, ha="right")
for i, v in enumerate(scores):
    h = H * v / scores.max()
    c = ACCENT if i == best else STEPS[1]
    ax.add_patch(mp.Rectangle((cx(i + (K - 1) / 2) - 0.12, yb), 0.24, h, fc=c, ec="none", zorder=3))

# ── 전역 max pooling ───────────────────────────────────────────────
bxc = cx(best + (K - 1) / 2)
arrow(ax, bxc + 0.18, 1.0, 7.0, 1.0, color=NEUTRAL)
label(ax, 5.85, 1.17, "전역 max pooling", fs=11, c=TEXT2)
box(ax, 7.05, 0.4, 1.3, 1.2, fc=FAINT, ec="none")
label(ax, 7.7, 1.38, "가장 센 반응", fs=11, c=TEXT2)
label(ax, 7.7, 1.13, "하나만 남김", fs=11, c=TEXT2)
label(ax, 7.7, 0.72, "위치는 버림", fs=12, w="bold")   # 본문에 없는 점수 숫자는 넣지 않음

save(fig, img("c10", "convolution.png"))

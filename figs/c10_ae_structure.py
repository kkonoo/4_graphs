"""10 §2-1 — 오토인코더의 구조: 입력 x → 인코더 → 좁은 잠재 z → 디코더 → 복원 x̂.

모래시계 모양. 손실 = ||x − x̂||². 칸 색은 설명용 가상의 값.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *

ENC, DEC = CAT[1], CAT[2]          # 인코더 · 디코더 (나란한 두 부분)
NEUTRAL = "#9AA1AB"

fig, ax = new(8.4, 3.55)

# ── 설명용 값: 입력 x 와 그것을 거의 되살린 복원 x̂ ─────────────────────────
rng = np.random.default_rng(7)
n_cell = 12
xv = rng.uniform(0.12, 0.9, n_cell)
xh = np.clip(xv + rng.normal(0, 0.08, n_cell), 0.08, 0.95)

y0, y1 = 0.75, 2.95                 # 입력·복원 칸 높이
cw = 0.36                           # 칸 폭
zc, zh = 1.85, 0.6                  # 병목 중심과 높이

def column(cx, vals):
    h = (y1 - y0) / len(vals)
    for i, a in enumerate(vals):
        box(ax, cx - cw / 2, y1 - (i + 1) * h, cw, h, fc=mix(PRIMARY, a), ec="white", lw=1.2, r=0.03, z=3)

column(0.9, xv)
column(7.5, xh)

# 잠재 z: 칸 3개 — 그림에서 놓치면 안 되는 한 곳(좁은 병목)
zh_cell = zh / 3
for i in range(3):
    box(ax, 4.2 - cw / 2, zc + zh / 2 - (i + 1) * zh_cell, cw, zh_cell, fc=ACCENT, ec="white", lw=1.2, r=0.03, z=3)

# ── 인코더 · 디코더 사다리꼴 (모래시계) ─────────────────────────────────────
ax.add_patch(mp.Polygon([(1.2, y0), (3.9, zc - zh / 2), (3.9, zc + zh / 2), (1.2, y1)],
                        closed=True, fc=mix(ENC, 0.16), ec="none", zorder=1))
ax.add_patch(mp.Polygon([(7.2, y0), (4.5, zc - zh / 2), (4.5, zc + zh / 2), (7.2, y1)],
                        closed=True, fc=mix(DEC, 0.16), ec="none", zorder=1))
label(ax, 2.3, 1.98, "인코더", fs=12.5, w="bold", c=shade(ENC, 0.2))
label(ax, 2.3, 1.68, "압축한다", fs=11, c=TEXT2)
label(ax, 6.1, 1.98, "디코더", fs=12.5, w="bold", c=shade(DEC, 0.2))
label(ax, 6.1, 1.68, "다시 펼친다", fs=11, c=TEXT2)

# ── 이름표 ─────────────────────────────────────────────────────────────────
label(ax, 0.9, 3.36, "입력 $x$", fs=12.5, w="bold")
label(ax, 0.9, 3.09, "유전자 1,838개", fs=11, c=TEXT2)
label(ax, 7.5, 3.36, "복원 $\\hat{x}$", fs=12.5, w="bold")
label(ax, 7.5, 3.09, "유전자 1,838개", fs=11, c=TEXT2)
label(ax, 4.2, 2.42, "잠재 $z$", fs=12.5, w="bold")
label(ax, 4.2, 1.3, "좁은 병목", fs=11, c=TEXT)

# ── 선형이면 PCA (작은 곁말) ───────────────────────────────────────────────
box(ax, 1.95, 2.98, 4.5, 0.4, fc=FAINT, ec="none", r=0.08, z=1)
label(ax, 4.2, 3.18, "활성함수를 빼면(선형 오토인코더) = PCA와 같은 부분공간", fs=11, c=TEXT2)

# ── 손실: 입력과 복원을 잇는 괄호 ───────────────────────────────────────────
yb = 0.42
ax.plot([0.9, 0.9, 7.5, 7.5], [y0 - 0.08, yb, yb, y0 - 0.08], color=NEUTRAL, lw=1.2, zorder=2,
        solid_joinstyle="round")
label(ax, 4.2, yb, "손실 = $\\Vert x - \\hat{x} \\Vert^2$", fs=12.5, w="bold", z=5,
      bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
label(ax, 4.2, 0.1, "입력과 복원의 차이만 줄인다 · 라벨이 필요 없다", fs=11, c=TEXT2)

print("설명용 그림: 입력 유전자 1,838개 → 인코더 → 잠재 z → 디코더 → 복원")
save(fig, img("c10", "ae_structure.png"))

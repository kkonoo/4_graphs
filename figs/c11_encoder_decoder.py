"""11 §1-2 — encoder–decoder 틀: 표 조회로 벡터를 꺼내고, 내적이 유사도를 재현하게 학습.

유사도의 정의(엣지 · 짧은 랜덤워크 · 구조의 모양)가 선택지라는 것을 아래 세 칸으로.
그래프와 Z의 값은 설명용.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *

NEUTRAL = "#9AA1AB"
V = PRIMARY
GE = shade(LINE, 0.1)          # 옅은 카드 위 엣지

fig, ax = new(8.3, 4.8)

def node(x, y, c, text=None, r=0.09, fs=11):
    ax.add_patch(mp.Circle((x, y), r, fc=c, ec="white", lw=1.2, zorder=4))
    if text:
        label(ax, x, y, text, fs=fs, w="bold", c=on(c), z=5)

def edge(p, q, c=LINE, lw=1.4, z=2):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, zorder=z, solid_capstyle="round")

# ── ① encoder–decoder ──────────────────────────────────────────────────────
label(ax, 0.05, 4.62, "① encoder–decoder", fs=12.5, w="bold", ha="left")
HY = 4.27
bx, bw = 6.35, 1.85
label(ax, 0.9, HY, "그래프", fs=11.5, c=TEXT2)
label(ax, 3.275, HY, "encoder · 표 조회", fs=11.5, w="bold")
label(ax, 5.075, HY, "임베딩", fs=11.5, c=TEXT2)
label(ax, bx + bw / 2, HY, "decoder · 내적", fs=11.5, w="bold")

# 작은 그래프
P = dict(u=(0.45, 3.55), a=(1.05, 3.82), v=(1.45, 3.2), b=(0.85, 3.05),
         c=(0.3, 2.72), d=(1.5, 2.6), e=(0.95, 2.5))
for s, t in [("u", "a"), ("u", "b"), ("a", "v"), ("b", "v"), ("c", "b"), ("u", "c"),
             ("b", "e"), ("v", "d"), ("d", "e")]:
    edge(P[s], P[t])
for k, p in P.items():
    if k in ("u", "v"):
        node(*p, V, k, r=0.13)
    else:
        node(*p, NEUTRAL)

arrow(ax, 1.8, 3.2, 2.3, 3.2, color=NEUTRAL)

# 표 Z (노드 수 × d)
tx, cw, rh, nrow, top = 2.65, 0.25, 0.21, 7, 3.97
vals = np.array([[.35, .60, .20, .80, .45], [.90, .30, .65, .20, .75], [.50, .25, .70, .40, .30],
                 [.20, .55, .35, .65, .50], [.70, .45, .85, .25, .60], [.40, .70, .30, .50, .20],
                 [.60, .30, .45, .75, .35]])
rows = {1: "u", 4: "v"}
for i in range(nrow):
    y0 = top - (i + 1) * rh
    for j in range(5):
        fc = mix(V, 0.2 + 0.75 * vals[i, j]) if i in rows else mix(NEUTRAL, 0.15 + 0.45 * vals[i, j])
        ax.add_patch(mp.Rectangle((tx + j * cw, y0), cw, rh, fc=fc, ec="white", lw=1.5, zorder=3))
    if i in rows:
        ax.add_patch(mp.Rectangle((tx, y0), 5 * cw, rh, fill=False, ec=V, lw=1.6, zorder=4))
        label(ax, tx - 0.13, y0 + rh / 2, rows[i], fs=11.5, w="bold", c=V)
label(ax, tx + 2.5 * cw, 2.32, "Z  (노드 수 × d)", fs=11, c=TEXT2)

# 꺼낸 벡터 z_u, z_v
sx, sw = 4.5, 0.23
for i, name in rows.items():
    y0 = top - (i + 1) * rh
    yc = y0 + rh / 2
    arrow(ax, tx + 5 * cw + 0.08, yc, sx - 0.08, yc, color=NEUTRAL, lw=1.4, ms=10)
    for j in range(5):
        ax.add_patch(mp.Rectangle((sx + j * sw, y0), sw, rh, fc=mix(V, 0.2 + 0.75 * vals[i, j]),
                                  ec="white", lw=1.5, zorder=3))
    label(ax, sx + 5 * sw + 0.08, yc, f"$z_{name}$", fs=13.5, c=V, ha="left")

# decoder
yu, yv = top - 1.5 * rh, top - 4.5 * rh
arrow(ax, 6.07, yu, 6.3, 3.36, color=NEUTRAL, lw=1.4, ms=10)
arrow(ax, 6.07, yv, 6.3, 3.27, color=NEUTRAL, lw=1.4, ms=10)
box(ax, bx, 2.5, bw, 1.47, fc=mix(V, 0.08), ec="none")
cx0 = bx + bw / 2
label(ax, cx0, 3.68, r"$z_u \cdot z_v$", fs=14, c=TEXT)
label(ax, cx0, 3.3, "≈ 유사도(u, v)", fs=13, w="bold", c=TEXT)
label(ax, cx0, 2.9, "내적이 유사도를 재현하도록", fs=11, c=TEXT2)
label(ax, cx0, 2.66, "Z를 학습한다", fs=11, c=TEXT2)

# ── ② 유사도의 정의 = 선택 ─────────────────────────────────────────────────
label(ax, 0.05, 2.0, "② “유사도”의 정의는 선택이다", fs=12.5, w="bold", ha="left")
opts = [("엣지가 있다", "바로 이웃이면 비슷하다", "라플라시안 고유맵"),
        ("짧은 랜덤워크에서 함께 나타난다", "몇 걸음 안에 자주 만나면 비슷하다", "DeepWalk, node2vec"),
        ("연결 구조의 모양이 같다", "멀리 있어도 같은 역할이면 비슷하다", "struc2vec")]
CW, CH, gy = 2.6, 1.78, 0.98
for k, ((title, mean, meth), x0) in enumerate(zip(opts, [0.1, 2.85, 5.6])):
    c = CAT[k]
    cx = x0 + CW / 2
    box(ax, x0, 0, CW, CH, fc=mix(c, 0.07), ec="none", r=0.1, z=1)
    label(ax, cx, 1.55, title, fs=11.5, w="bold", c=c)
    label(ax, cx, 0.44, mean, fs=11, c=TEXT)
    label(ax, cx, 0.17, f"대표: {meth}", fs=11, c=TEXT2)
    if k == 0:                                   # 바로 이웃
        u, v = (cx - 0.32, gy), (cx + 0.32, gy)
        g = [(cx - 0.8, gy + 0.25), (cx - 0.78, gy - 0.25), (cx + 0.8, gy + 0.22), (cx + 0.75, gy - 0.25)]
        for p in g[:2]: edge(u, p, c=GE)
        for p in g[2:]: edge(v, p, c=GE)
        edge(g[0], g[1], c=GE)
        edge(u, v, c=c, lw=2.6, z=3)
        for p in g: node(*p, NEUTRAL)
    elif k == 1:                                 # 짧은 랜덤워크
        u, a, b, v = (cx - 0.85, gy - 0.12), (cx - 0.3, gy + 0.22), (cx + 0.25, gy - 0.15), (cx + 0.8, gy + 0.15)
        gg, hh = (cx - 0.3, gy - 0.33), (cx + 0.32, gy + 0.33)
        for s, t in [(u, a), (a, b), (b, v), (u, gg), (gg, b), (a, hh), (hh, v)]:
            edge(s, t, c=GE)
        for s, t in [(u, a), (a, b), (b, v)]:
            d = np.array(t) - np.array(s); d = d / np.linalg.norm(d) * 0.15
            arrow(ax, s[0] + d[0], s[1] + d[1], t[0] - d[0], t[1] - d[1], color=c, lw=1.4, ms=9,
                  rad=-0.35, z=3)
        for p in (a, b, gg, hh): node(*p, NEUTRAL)
    else:                                        # 같은 모양, 먼 곳
        u, v = (cx - 0.68, gy), (cx + 0.68, gy)
        leaves = {}
        for nm, ctr in (("u", u), ("v", v)):
            ax.add_patch(mp.Circle(ctr, 0.38, fc="none", ec=mix(c, 0.6), lw=1.1, ls=(0, (3, 2)), zorder=1))
            for ang in (90, 210, 330):
                a_ = np.deg2rad(ang)
                p = (ctr[0] + 0.27 * np.cos(a_), ctr[1] + 0.27 * np.sin(a_))
                leaves[(nm, ang)] = p
                edge(ctr, p, c=GE)
        m = (cx, gy - 0.3)
        edge(leaves[("u", 330)], m, c=GE); edge(m, leaves[("v", 210)], c=GE)
        for p in list(leaves.values()) + [m]: node(*p, NEUTRAL, r=0.08)
    node(*u, c, "u", r=0.12)
    node(*v, c, "v", r=0.12)

print("설명용 개념도 — 숫자 없음 (유사도 정의 3가지: 엣지 / 짧은 랜덤워크 / 구조의 모양)")
save(fig, img("c11", "encoder_decoder.png"))

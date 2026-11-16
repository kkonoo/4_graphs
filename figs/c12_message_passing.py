"""12 §1-2 — 메시지 전달: 한 층에서 노드 v가 하는 일, 층을 쌓으면 보는 범위.

본문의 식 h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u ∈ N(v)}))를 그림으로 옮김.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *

V, NB, HOP2 = PRIMARY, CAT[1], CAT[2]          # 나 · 1-hop 이웃 · 2-hop 이웃 (두 패널에서 같은 뜻 = 같은 색)
R = 0.17                                        # 노드 반지름 (인치)

fig, ax = new(8.4, 3.45)

def node(x, y, c, text=None, r=R, fs=11.5):
    ax.add_patch(mp.Circle((x, y), r, fc=c, ec="white", lw=1.2, zorder=4))
    if text:
        label(ax, x, y, text, fs=fs, w="bold", c=on(c), z=5)

def edge(p, q, c=LINE, lw=1.4, z=2):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, zorder=z, solid_capstyle="round")

# ── ① 한 층 ────────────────────────────────────────────────────────────────
label(ax, 0.05, 3.3, "① 한 층에서 노드 v가 하는 일", fs=12.5, w="bold", ha="left")
v = (1.15, 1.25)
us = [(0.35, 2.3), (1.25, 2.55), (0.3, 1.0)]
for u in us:
    edge(u, v)
for u in us:
    node(*u, NB, "u")
node(*v, V, "v", r=0.21)
label(ax, 0.8, 2.85, "이웃 N(v)", fs=11, c=TEXT2)

# 흐름: 이웃 → AGGREGATE → UPDATE ← 나 → 새 표현
bx, bw = 2.45, 2.95
box(ax, bx, 2.05, bw, 0.82, fc=mix(NB, 0.12), ec="none")
label(ax, bx + 0.15, 2.62, "AGGREGATE · 이웃의 표현을 모은다", fs=11.5, w="bold", ha="left")
label(ax, bx + 0.15, 2.3, "평균 · 합 · 어텐션 가중 평균 (순서와 무관)", fs=11, c=TEXT2, ha="left")
box(ax, bx, 0.95, bw, 0.82, fc=mix(V, 0.12), ec="none")
label(ax, bx + 0.15, 1.52, "UPDATE · 내 표현과 합쳐 W를 곱하고 σ", fs=11.5, w="bold", ha="left")
label(ax, bx + 0.15, 1.2, "W는 모든 노드가 같은 것을 쓴다", fs=11, c=TEXT2, ha="left")

arrow(ax, 1.5, 2.46, bx - 0.06, 2.46, color=NB)
label(ax, 1.95, 2.62, "$h_u^{(k)}$", fs=12, c=NB)
arrow(ax, v[0] + 0.25, v[1] + 0.05, bx - 0.06, 1.3, color=V)
label(ax, 1.95, 1.0, "$h_v^{(k)}$", fs=12, c=V)
arrow(ax, bx + bw / 2, 2.05, bx + bw / 2, 1.8, color="#9AA1AB")
arrow(ax, bx + bw / 2, 0.95, bx + bw / 2, 0.55, color="#9AA1AB")
pill(ax, bx + bw / 2, 0.33, "새 표현  $h_v^{(k+1)}$", fc=V, fs=11.5, h=0.36)

# ── ② 층을 쌓으면 ──────────────────────────────────────────────────────────
cx, cy = 6.98, 1.45
label(ax, 5.8, 3.3, "② 층 수 = 보는 거리", fs=12.5, w="bold", ha="left")
for rr, c in [(1.08, HOP2), (0.58, NB)]:
    ax.add_patch(mp.Circle((cx, cy), rr, fc=mix(c, 0.07), ec=mix(c, 0.55), lw=1.1, ls=(0, (4, 3)), zorder=1))
a1 = np.deg2rad([95, 215, 330])
p1 = [(cx + 0.58 * np.cos(a), cy + 0.58 * np.sin(a)) for a in a1]
a2 = np.deg2rad([[70, 120], [195, 245], [300, 355]])
p2 = [[(cx + 1.08 * np.cos(a), cy + 1.08 * np.sin(a)) for a in pair] for pair in a2]
for q, pair in zip(p1, p2):
    edge((cx, cy), q)
    for s in pair:
        edge(q, s)
for pair in p2:
    for s in pair:
        node(*s, HOP2, r=0.13)
for q in p1:
    node(*q, NB, r=0.15)
node(cx, cy, V, "v", r=0.19)
a = np.deg2rad(157)
label(ax, cx + 0.36 * np.cos(a), cy + 0.36 * np.sin(a), "1층", fs=11, w="bold", c=shade(NB, 0.2))
label(ax, cx + 0.83 * np.cos(a), cy + 0.83 * np.sin(a), "2층", fs=11, w="bold", c=shade(HOP2, 0.2))
label(ax, cx, 0.02, "k층을 쌓으면 k-hop 이웃까지 본다", fs=11, c=TEXT, va="bottom")

save(fig, img("c12", "message_passing.png"))

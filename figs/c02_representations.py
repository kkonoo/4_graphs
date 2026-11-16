"""02 §2 — 같은 그래프, 세 가지 표현.

그림 / 인접 행렬 / 엣지 목록·인접 리스트.
위: ① 그림 · ② 인접 행렬,  아래: ③ 엣지 목록 · ④ 인접 리스트.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx

names = ["TP53", "MDM2", "CDKN1A", "ATM", "CHEK2", "BAX", "RB1", "E2F1"]
edges = [("TP53", "MDM2"), ("TP53", "CDKN1A"), ("TP53", "BAX"), ("TP53", "ATM"),
         ("ATM", "CHEK2"), ("CHEK2", "TP53"), ("CDKN1A", "RB1"), ("RB1", "E2F1")]
G = nx.Graph(); G.add_nodes_from(names); G.add_edges_from(edges)

A = nx.to_numpy_array(G, nodelist=names)
edge_rows = sorted(G.edges())
adj_rows = [(n, ", ".join(sorted(G[n]))) for n in names]

NODE = PRIMARY
ZERO_TXT = "#B3B8C0"

fig, ax = new(8.4, 6.4)

# ── ① 그림 ────────────────────────────────────────────────────────────────
label(ax, 0.05, 6.25, "① 그림", fs=12.5, w="bold", ha="left")
# 손으로 배치한 자리 (노드 이름이 겹치지 않도록). 연결 관계는 G 그대로.
pos = {"TP53": (1.75, 4.5), "MDM2": (0.75, 5.5), "BAX": (0.6, 4.3), "ATM": (0.95, 3.3),
       "CHEK2": (2.15, 3.3), "CDKN1A": (2.95, 5.3), "RB1": (3.5, 4.3), "E2F1": (3.5, 3.3)}
for u, v in G.edges():
    (x1, y1), (x2, y2) = pos[u], pos[v]
    ax.plot([x1, x2], [y1, y2], color=LINE, lw=2.2, zorder=2, solid_capstyle="round")
for n in names:
    pill(ax, *pos[n], n, fc=NODE, fs=11.5, h=0.36, padx=0.14)
label(ax, 2.05, 2.72, f"노드 {G.number_of_nodes()}개 · 엣지 {G.number_of_edges()}개", fs=11, c=TEXT2)

# ── ② 인접 행렬 ───────────────────────────────────────────────────────────
c = 0.3
x0, y_top = 5.6, 5.3
k = len(names)
label(ax, 4.75, 6.25, f"② 인접 행렬 — 칸 {k*k}개 중 {int(A.sum())}개만 1", fs=12.5, w="bold", ha="left")
for i in range(k):
    for j in range(k):
        x, y = x0 + j * c, y_top - (i + 1) * c
        one = A[i, j] > 0
        box(ax, x + 0.02, y + 0.02, c - 0.04, c - 0.04, fc=NODE if one else FAINT, ec="none", r=0.03)
        label(ax, x + c / 2, y + c / 2, str(int(A[i, j])), fs=11, w="bold" if one else "regular",
              c="white" if one else ZERO_TXT)
for i, n in enumerate(names):
    label(ax, x0 - 0.08, y_top - (i + 0.5) * c, n, fs=11, ha="right", c=TEXT)
    label(ax, x0 + (i + 0.5) * c, y_top + 0.07, n, fs=11, va="bottom", c=TEXT, rotation=90)
label(ax, x0 + k * c / 2, 2.72, "엣지 하나가 두 칸 (행렬이 대칭)", fs=11, c=TEXT2)

# ── ③ 엣지 목록 ───────────────────────────────────────────────────────────
y_mark, y_head, y0, dy = 2.3, 1.98, 1.7, 0.25
label(ax, 0.05, y_mark, f"③ 엣지 목록 — {len(edge_rows)}줄", fs=12.5, w="bold", ha="left")
label(ax, 0.25, y_head, "노드 1", fs=11, c=TEXT2, ha="left")
label(ax, 1.55, y_head, "노드 2", fs=11, c=TEXT2, ha="left")
for r, (u, v) in enumerate(edge_rows):
    y = y0 - r * dy
    if r % 2 == 0:
        box(ax, 0.1, y - dy / 2, 3.2, dy, fc=FAINT, ec="none", r=0.04, z=1)
    label(ax, 0.25, y, u, fs=11, ha="left")
    label(ax, 1.55, y, v, fs=11, ha="left")

# ── ④ 인접 리스트 ─────────────────────────────────────────────────────────
xa = 4.2
label(ax, xa - 0.15, y_mark, "④ 인접 리스트 — 노드마다 한 줄", fs=12.5, w="bold", ha="left")
label(ax, xa + 0.05, y_head, "노드", fs=11, c=TEXT2, ha="left")
label(ax, xa + 1.2, y_head, "이웃", fs=11, c=TEXT2, ha="left")
for r, (n, nb) in enumerate(adj_rows):
    y = y0 - r * dy
    if r % 2 == 0:
        box(ax, xa - 0.1, y - dy / 2, 4.2, dy, fc=FAINT, ec="none", r=0.04, z=1)
    label(ax, xa + 0.05, y, n, fs=11, w="bold", ha="left")
    label(ax, xa + 1.2, y, nb, fs=11, ha="left")

print(f"노드 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개")
print(f"인접 행렬: 칸 {k*k}개 중 {int(A.sum())}개만 1")
print(f"엣지 목록: {len(edge_rows)}줄")
for u, v in edge_rows:
    print(f"  {u:<8}{v}")
print("인접 리스트:")
for n, nb in adj_rows:
    print(f"  {n:<8}{nb}")

save(fig, img("c02", "representations.png"))

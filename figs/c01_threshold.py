"""01 §3 — 임계값이 네트워크를 정한다.

같은 상관 행렬, 세 개의 기준값. 위: 기준값에 따른 엣지 수, 아래: 세 기준값에서의 네트워크.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter

rng = np.random.default_rng(4)
nmod, per = 4, 12
n = nmod * per
lab = np.repeat(np.arange(nmod), per)

C = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        base = 0.70 if lab[i] == lab[j] else 0.22
        C[i, j] = C[j, i] = np.clip(base + rng.normal(0, 0.15), -0.2, 0.99)

ths = np.arange(0.30, 0.951, 0.01)
counts = [(C > t).sum() // 2 for t in ths]

NEUTRAL = "#9AA1AB"
MARK = ["①", "②", "③"]
TH = [0.50, 0.60, 0.75]
VERDICT = ["모듈이 맞닿아 경계가 불분명", "깔끔한 네 개의 모듈", "모듈이 조각남"]

fig = plt.figure(figsize=(8.4, 6.2))

# ── 위: 기준값에 따른 엣지 수 ─────────────────────────────────────────────
ax = fig.add_axes([0.11, 0.635, 0.84, 0.325])
ax.plot(ths, counts, color=PRIMARY, lw=2.2, zorder=3)
for t in TH:
    ax.axvline(t, color=NEUTRAL, ls=(0, (4, 3)), lw=1.1, zorder=1)
for t, txt in [(0.30, None), (0.50, "①"), (0.60, "②"), (0.75, "③"), (0.90, None)]:
    e = (C > t).sum() // 2
    print(f"r > {t:.2f}  엣지 {e}개")
    main = txt is not None
    ax.scatter([t], [e], s=40 if main else 30, color=TEXT if main else NEUTRAL, zorder=5, lw=0)
    s = f"{txt} {e}" if main else f"{e}"
    ax.annotate(s, (t, e), xytext=(7, 4), textcoords="offset points", ha="left", va="bottom",
                fontsize=11.5 if main else 11, fontweight="bold" if main else "regular",
                color=TEXT if main else TEXT2, zorder=6)
ax.set_yscale("log")
ax.set_ylim(10, 900)
ax.yaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 500]))
ax.yaxis.set_major_formatter(FixedFormatter(["10", "20", "50", "100", "200", "500"]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlim(0.28, 0.97)
ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax.tick_params(labelsize=11)
ax.grid(axis="y", color=LINE, lw=0.8, zorder=0)
ax.set_xlabel("상관 임계값  r", fontsize=11.5)
ax.set_ylabel("엣지 수", fontsize=11.5)

# ── 아래: 세 기준값의 네트워크 ───────────────────────────────────────────
cols = [CAT[l] for l in lab]             # 실제 모듈 4개 (나란한 범주)
for idx, t in enumerate(TH):
    G = nx.Graph((i, j) for i in range(n) for j in range(i + 1, n) if C[i, j] > t)
    G.add_nodes_from(range(n))
    pos = nx.spring_layout(G, seed=2, k=0.55)
    a = fig.add_axes([0.01 + idx * 0.333, 0.075, 0.315, 0.40])
    nx.draw_networkx_edges(G, pos, ax=a, edge_color=LINE, width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=a, nodelist=list(range(n)), node_color=cols,
                           linewidths=0, node_size=36)
    a.margins(0.06); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)
    a.text(0.5, 1.03, f"{MARK[idx]} r > {t:.2f} · 엣지 {G.number_of_edges()}개", transform=a.transAxes,
           ha="center", va="bottom", fontsize=12, fontweight="bold", color=TEXT)
    a.text(0.5, -0.03, VERDICT[idx], transform=a.transAxes, ha="center", va="top",
           fontsize=11, color=TEXT2)

save(fig, img("c01", "threshold.png"))

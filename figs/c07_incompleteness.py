"""07 §2 — 네트워크가 불완전할 때 순위는 얼마나 흔들리는가.

엣지의 일부만 관측했다고 치고, 전체를 봤을 때의 순위와 비교한다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
import pandas as pd
from scipy.stats import spearmanr

rng = np.random.default_rng(0)

G = nx.powerlaw_cluster_graph(600, 3, 0.3, seed=2)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
nodes = list(G.nodes())
full_deg = pd.Series(dict(G.degree()))
full_btw = pd.Series(nx.betweenness_centrality(G))
edges = list(G.edges()); m = len(edges)

fracs = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
res = {k: [] for k in ("deg_rho", "btw_rho", "deg_ov", "btw_ov")}
for frac in fracs:
    d_r, b_r, d_o, b_o = [], [], [], []
    for rep in range(10):
        keep = rng.choice(m, int(frac * m), replace=False)
        H = nx.Graph(); H.add_nodes_from(nodes)
        H.add_edges_from([edges[i] for i in keep])
        d = pd.Series(dict(H.degree()))
        b = pd.Series(nx.betweenness_centrality(H))
        d_r.append(spearmanr(d[nodes].values, full_deg[nodes].values).statistic)
        b_r.append(spearmanr(b[nodes].values, full_btw[nodes].values).statistic)
        d_o.append(len(set(d.nlargest(20).index) & set(full_deg.nlargest(20).index)))
        b_o.append(len(set(b.nlargest(20).index) & set(full_btw.nlargest(20).index)))
    res["deg_rho"].append(np.mean(d_r)); res["btw_rho"].append(np.mean(b_r))
    res["deg_ov"].append(np.mean(d_o));  res["btw_ov"].append(np.mean(b_o))
    print(f"관측 {frac:.0%}: 차수 rho {np.mean(d_r):.3f} 상위20 {np.mean(d_o):.1f}/20 | "
          f"매개 rho {np.mean(b_r):.3f} 상위20 {np.mean(b_o):.1f}/20")

# ── drawing ────────────────────────────────────────────────────────────────
NEUTRAL = "#9AA1AB"
DC, BC = CAT[0], CAT[2]                    # 차수 · 매개 (나란한 두 지표)
x = np.array(fracs) * 100
plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"wspace": 0.34})

panels = [(axes[0], "① 순위 전체", "deg_rho", "btw_rho", "전체 순위와의 스피어만 상관 ρ", (0.4, 1.02)),
          (axes[1], "② 상위 20개", "deg_ov", "btw_ov", "상위 20개 중 유지된 수", (8, 20.5))]
for ax, mark, dk, bk, ylab, ylim in panels:
    lo, hi = ylim
    ax.axvspan(45, 55, color=FAINT, lw=0, zorder=0)
    ax.text(50, hi - 0.035 * (hi - lo), "절반 관측", ha="center", va="top", fontsize=11, color=TEXT2)
    ax.plot(x, res[dk], "o-", color=DC, lw=2, ms=5, zorder=3)
    ax.plot(x, res[bk], "s-", color=BC, lw=2, ms=5, zorder=3)
    ax.text(27.5, res[dk][-1], "차수", color=DC, fontsize=11.5, fontweight="bold", va="center", ha="left")
    ax.text(27.5, res[bk][-1], "매개", color=BC, fontsize=11.5, fontweight="bold", va="center", ha="left")
    ax.set_xlim(98, 18)                    # 뒤집힌 축: 오른쪽 끝에 직접 라벨 자리
    ax.set_xticks([90, 80, 70, 60, 50, 40, 30])
    ax.set_ylim(lo, hi)
    ax.set_xlabel("관측된 엣지 비율 (%)", fontsize=11.5)
    ax.set_ylabel(ylab, fontsize=11.5)
    ax.text(0.0, 1.05, mark, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12.5, fontweight="bold", color=TEXT)

ax = axes[0]
ax.axhline(0.8, color=NEUTRAL, ls=(0, (4, 3)), lw=1.2, zorder=1)
ax.text(36, 0.815, "ρ = 0.8", color=TEXT2, fontsize=11, ha="center", va="bottom")

save(fig, img("c07", "incompleteness.png"))

"""07 §2 — 네트워크가 불완전할 때 순위는 얼마나 흔들리는가.

엣지의 일부만 관측했다고 치고, 전체를 봤을 때의 순위와 비교한다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx
import pandas as pd
from scipy.stats import spearmanr

OUT = outdir("c07")
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

x = np.array(fracs) * 100
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), gridspec_kw={"wspace": 0.24})

ax = axes[0]
ax.plot(x, res["deg_rho"], "o-", color=TEAL, lw=2, ms=5, label="차수 중심성")
ax.plot(x, res["btw_rho"], "s-", color=ORANGE, lw=2, ms=5, label="매개 중심성")
ax.axhline(0.8, color=RED, ls="--", lw=1.3)
ax.text(32, 0.815, "ρ = 0.8", color=RED, fontsize=10)
ax.set_xlabel("관측된 엣지 비율 (%)"); ax.set_ylabel("전체 순위와의 스피어만 상관 ρ")
ax.set_ylim(0.4, 1.02); ax.invert_xaxis()
ax.set_title("순위 전체의 재현성", color=NAVY, fontsize=12)
ax.legend(frameon=False, fontsize=10)

ax = axes[1]
ax.plot(x, res["deg_ov"], "o-", color=TEAL, lw=2, ms=5, label="차수 중심성")
ax.plot(x, res["btw_ov"], "s-", color=ORANGE, lw=2, ms=5, label="매개 중심성")
ax.set_xlabel("관측된 엣지 비율 (%)"); ax.set_ylabel("상위 20개 중 유지된 수")
ax.set_ylim(8, 20.5); ax.invert_xaxis()
ax.set_title("상위 20개는 비교적 버틴다", color=NAVY, fontsize=12)
ax.legend(frameon=False, fontsize=10)

fig.suptitle("엣지의 일부만 봤을 때 — 상위 몇 개는 살아남지만 순위 전체는 무너진다",
             color=NAVY, fontsize=13, fontweight="bold", y=1.02)
fig.savefig(os.path.join(OUT, "incompleteness.png"))
print("wrote", os.path.join(OUT, "incompleteness.png"))

"""01 §2: 네 가지 생물학 네트워크, 하나의 자료구조.

위 둘: 연결이 측정되었다. 아래 둘: 우리가 만들었고 파라미터가 모양을 정했다.
(계산은 이전 스크립트와 같고, 모양만 새 스타일. 1×4 → 2×2로 쌓음)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from scipy.spatial import cKDTree

NEUTRAL = "#9AA1AB"
plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.5), gridspec_kw={"wspace": 0.08, "hspace": 0.42})
axes = axes.ravel()
rng = np.random.default_rng(11)

def frame(ax, marker, note):
    ax.margins(0.05); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(0.5, 1.03, marker, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12.5, fontweight="bold", color=TEXT)
    ax.text(0.5, -0.03, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color=TEXT2)

# --- 1. PPI: 허브가 있는 구조 ----------------------------------------------
G = nx.barabasi_albert_graph(70, 2, seed=3)
pos = nx.spring_layout(G, seed=3, k=0.42)
deg = np.array([d for _, d in G.degree()])
ax = axes[0]
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LINE, width=0.7)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=CAT[0], linewidths=0,
                       node_size=18 + 9 * deg)
frame(ax, "① 단백질 상호작용 (PPI)", "측정된 엣지 (Y2H, AP-MS, 큐레이션)")

# --- 2. 대사·신호전달: 방향이 있다 -----------------------------------------
D = nx.DiGraph()
for ch in (list(range(0, 7)), list(range(7, 13)), list(range(13, 19))):
    D.add_edges_from(zip(ch[:-1], ch[1:]))
D.add_edges_from([(3, 8), (9, 15), (16, 5), (1, 14)])
posd = nx.kamada_kawai_layout(D)
ax = axes[1]
nx.draw_networkx_edges(D, posd, ax=ax, edge_color=NEUTRAL, width=1.1,
                       arrowsize=10, node_size=110)
nx.draw_networkx_nodes(D, posd, ax=ax, node_color=CAT[0], linewidths=0,
                       node_size=110)
frame(ax, "② 대사 · 신호전달", "측정된 엣지 · 방향과 부호가 있다")

# --- 3. 공발현: 상관 임계값을 우리가 골랐다 --------------------------------
nmod, per = 4, 12
lab = np.repeat(np.arange(nmod), per)
n = nmod * per
C = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        base = 0.72 if lab[i] == lab[j] else 0.18
        C[i, j] = C[j, i] = np.clip(base + rng.normal(0, 0.13), 0, 0.99)
Gc = nx.Graph((i, j) for i in range(n) for j in range(i + 1, n) if C[i, j] > 0.60)
Gc.add_nodes_from(range(n))
posc = nx.spring_layout(Gc, seed=7, k=0.5)
cols = [CAT[l] for l in lab]
ax = axes[2]
nx.draw_networkx_edges(Gc, posc, ax=ax, edge_color=LINE, width=0.7)
nx.draw_networkx_nodes(Gc, posc, ax=ax, node_color=cols, linewidths=0,
                       node_size=55)
frame(ax, "③ 유전자 공발현", "만든 엣지 · 우리가 r > 0.60으로 정했다")

# --- 4. 세포 kNN 그래프: k 를 우리가 골랐다 --------------------------------
pops = [(34, (-1.7, 1.5), 0.55, CAT[0]), (28, (1.6, 1.7), 0.50, CAT[1]),
        (24, (1.1, -1.5), 0.48, CAT[2]), (14, (-1.9, -1.6), 0.40, CAT[3])]
X, cc = [], []
for cnt, (cx, cy), s, col in pops:
    X.append(rng.normal([cx, cy], s, size=(cnt, 2))); cc += [col] * cnt
X = np.vstack(X)
tree = cKDTree(X); k = 5
Gk = nx.Graph(); Gk.add_nodes_from(range(len(X)))
for i, nb in enumerate(tree.query(X, k + 1)[1]):
    for j in nb[1:]:
        Gk.add_edge(i, int(j))
posk = {i: tuple(X[i]) for i in range(len(X))}
ax = axes[3]
nx.draw_networkx_edges(Gk, posk, ax=ax, edge_color=LINE, width=0.7)
nx.draw_networkx_nodes(Gk, posk, ax=ax, node_color=cc, linewidths=0, node_size=45)
frame(ax, "④ 단일세포 kNN 그래프", "만든 엣지 · 우리가 k = 5로 정했다")

print(f"PPI: 노드 {G.number_of_nodes()}  엣지 {G.number_of_edges()}")
print(f"대사·신호전달: 노드 {D.number_of_nodes()}  엣지 {D.number_of_edges()}")
print(f"공발현 (r > 0.60): 노드 {Gc.number_of_nodes()}  엣지 {Gc.number_of_edges()}")
print(f"kNN (k = {k}): 노드 {Gk.number_of_nodes()}  엣지 {Gk.number_of_edges()}")
save(fig, img("c01", "bio_networks.png"))

"""11 — 노드 임베딩 실측: node2vec의 p, q / 행렬 분해와의 비교 / 시드와 좌표.

그래프: 05 §2와 같은 확률 블록 모형 (커뮤니티 6개 x 80노드, 안쪽 연결 확률 0.10,
사이 연결 확률 0.010 / 0.018 / 0.026, nx.stochastic_block_model(seed=1))
node2vec: 노드마다 걸음 10개 x 길이 40, word2vec skip-gram (창 5, 차원 32, 음성 5,
epoch 5, workers=1, seed 고정). p = q = 1이면 DeepWalk와 같음.

실험 1 (c11_walk_reach.csv): 평면 위 무작위 점 600개의 kNN 그래프(k = 6, 대칭화; 공간 전사체의
         이웃 그래프와 같은 모양)에서 걸음이 t걸음 뒤 출발점에서 몇 hop 떨어져 있나 (p = 1, 걸음 5 x 노드)
실험 2 (c11_sbm_compare.csv): 정답 커뮤니티와의 ARI — Leiden, 스펙트럴, NetMF, DeepWalk, node2vec
실험 3 (c11_lesmis.csv): node2vec 논문의 Les Misérables 설정 (d = 16, 걸음 10 x 80, 창 10,
         1 epoch, p = 1, q = 0.5 vs 2) — k-means 6개 군집이 커뮤니티(Leiden)를 따르나, 차수를 따르나
실험 4 (c11_seed.csv): 시드만 바꾼 두 DeepWalk 임베딩 — 좌표 vs 노드 쌍 유사도
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import pandas as pd
import networkx as nx
import igraph as ig, leidenalg as la
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

OUT = outdir("c11")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
P_OUTS = [0.010, 0.018, 0.026]


def spatial_knn(n=600, k=6, seed=0):
    from sklearn.neighbors import kneighbors_graph
    P = np.random.default_rng(seed).random((n, 2))
    A = kneighbors_graph(P, k, include_self=False)
    return nx.from_scipy_sparse_array(((A + A.T) > 0).astype(int)), P


def sbm(p_out):
    P = np.full((6, 6), p_out); np.fill_diagonal(P, 0.10)
    return nx.stochastic_block_model([80] * 6, P, seed=1), np.repeat(np.arange(6), 80)


def walks(G, p=1.0, q=1.0, n_walks=10, length=40, seed=0):
    """node2vec의 2차 랜덤워크. 직전 노드로 돌아가기 1/p, 직전 노드의 이웃 1, 그 밖 1/q"""
    rng = np.random.default_rng(seed)
    nbrs = {v: list(G.neighbors(v)) for v in G}
    nset = {v: set(nbrs[v]) for v in G}
    out = []
    for _ in range(n_walks):
        for s in rng.permutation(list(G)):
            w = [s]
            while len(w) < length:
                nb = nbrs[w[-1]]
                if len(w) == 1:
                    w.append(nb[rng.integers(len(nb))]); continue
                prev = w[-2]
                wt = np.array([1 / p if x == prev else (1.0 if x in nset[prev] else 1 / q) for x in nb])
                w.append(nb[rng.choice(len(nb), p=wt / wt.sum())])
            out.append(w)
    return out


def node2vec(G, p=1.0, q=1.0, dim=32, window=5, epochs=5, seed=0, **kw):
    W = [[str(x) for x in w] for w in walks(G, p, q, seed=seed, **kw)]
    m = Word2Vec(W, vector_size=dim, window=window, min_count=0, sg=1, negative=5,
                 epochs=epochs, workers=1, seed=seed)
    return np.array([m.wv[str(v)] for v in G])


def netmf(G, T=5, b=1, dim=32):
    """Qiu et al. 2018: DeepWalk가 암묵적으로 분해하는 행렬을 직접 만들어 SVD"""
    A = nx.to_numpy_array(G); deg = A.sum(1); P = A / deg[:, None]
    S, Pr = np.zeros_like(A), np.eye(len(A))
    for _ in range(T):
        Pr = Pr @ P; S += Pr
    M = A.sum() / (b * T) * S / deg[None, :]
    U, s, _ = np.linalg.svd(np.log(np.maximum(M, 1)))
    return U[:, :dim] * np.sqrt(s[:dim])


def spectral(G, k):
    w, V = np.linalg.eigh(nx.normalized_laplacian_matrix(G).toarray())
    return V[:, :k]


def km(Z, k):
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return KMeans(k, n_init=10, random_state=0).fit_predict(Z)


def leiden(G, seed=0):
    g = ig.Graph.from_networkx(G)
    return np.array(la.find_partition(g, la.RBConfigurationVertexPartition,
                                      resolution_parameter=1.0, seed=seed).membership)


def eta2(y, lab):
    m = y.mean()
    return sum(((y[lab == c].mean() - m) ** 2) * (lab == c).sum() for c in set(lab)) / ((y - m) ** 2).sum()


if __name__ == "__main__":
    f1, f2, f3, f4 = [os.path.join(CACHE, f"c11_{n}.csv") for n in
                      ("walk_reach", "sbm_compare", "lesmis", "seed")]
    if not os.path.exists(f1):
        G, _ = spatial_knn()
        D = dict(nx.all_pairs_shortest_path_length(G))
        rows = []
        for q in [0.25, 0.5, 1, 2, 4]:
            W = np.array(walks(G, 1, q, n_walks=5, length=21, seed=0))
            for t in range(1, 21):
                hop = np.array([D[w[0]][w[t]] for w in W])
                rows.append(dict(q=q, t=t, hop=hop.mean(), hop_sd=hop.std()))
        pd.DataFrame(rows).to_csv(f1, index=False)
        print(pd.DataFrame(rows).query("t in [1, 5, 10, 20]").round(3), flush=True)
    if not os.path.exists(f2):
        rows = []
        for po in P_OUTS:
            G, truth = sbm(po)
            for s in range(5):
                rows.append(dict(p_out=po, method="Leiden", seed=s, ari=ari(truth, leiden(G, s))))
            rows.append(dict(p_out=po, method="스펙트럴", seed=0, ari=ari(truth, km(spectral(G, 6), 6))))
            rows.append(dict(p_out=po, method="NetMF", seed=0, ari=ari(truth, km(netmf(G), 6))))
            for name, p, q in [("DeepWalk", 1, 1), ("node2vec q=0.5", 1, 0.5), ("node2vec q=2", 1, 2)]:
                for s in range(5):
                    t = time.time()
                    rows.append(dict(p_out=po, method=name, seed=s,
                                     ari=ari(truth, km(node2vec(G, p, q, seed=s), 6))))
                    print(po, name, s, round(rows[-1]["ari"], 3), f"{time.time()-t:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(f2, index=False)
    if not os.path.exists(f3):
        G = nx.Graph(nx.les_miserables_graph())
        for u, v in G.edges():
            G[u][v].pop("weight", None)                    # 가중치는 쓰지 않음
        comm = leiden(G, 0)
        logdeg = np.log(np.array([G.degree(v) for v in G]))
        rows = []
        for q in [0.5, 2]:
            for s in range(10):
                Z = node2vec(G, 1, q, dim=16, window=10, epochs=1, seed=s, n_walks=10, length=80)
                lab = KMeans(6, n_init=10, random_state=0).fit_predict(Z)
                rows.append(dict(q=q, seed=s, ari_comm=ari(comm, lab), eta2_deg=eta2(logdeg, lab)))
        pd.DataFrame(rows).to_csv(f3, index=False)
        print(pd.DataFrame(rows).groupby("q").mean().round(3), flush=True)
    if not os.path.exists(f4):
        G, truth = sbm(0.018)
        Za, Zb = node2vec(G, seed=0), node2vec(G, seed=1)
        coord = np.mean([abs(np.corrcoef(Za[:, j], Zb[:, j])[0, 1]) for j in range(Za.shape[1])])
        cos = lambda Z: (Z / np.linalg.norm(Z, axis=1, keepdims=True)) @ (Z / np.linalg.norm(Z, axis=1, keepdims=True)).T
        iu = np.triu_indices(len(Za), 1)
        sim = np.corrcoef(cos(Za)[iu], cos(Zb)[iu])[0, 1]
        agree = ari(km(Za, 6), km(Zb, 6))
        pd.DataFrame([dict(coord_abs_r=coord, pair_cos_r=sim, kmeans_ari=agree)]).to_csv(f4, index=False)
        print("seed:", round(coord, 3), round(sim, 3), round(agree, 3), flush=True)

    # ── 그림 1: q가 걸음을 바꾼다 ─────────────────────────────────────
    r = pd.read_csv(f1)
    G, P = spatial_knn()
    start = int(np.argmin(((P - 0.5) ** 2).sum(1)))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, q, c in [(axes[0], 0.25, PURPLE), (axes[1], 4, TEAL)]:
        for u, v in G.edges():
            ax.plot(*P[[u, v]].T, color=LGREY, lw=0.6, zorder=1)
        ax.scatter(*P.T, s=5, color=GREY, zorder=2)
        rng = np.random.default_rng(1)
        for k in range(3):                                # 같은 출발점에서 20걸음 x 3번
            w = walks(G.subgraph(G.nodes), 1, q, n_walks=1, length=21, seed=10 + k)
            w = [x for x in w if x[0] == start][0]
            ax.plot(*P[w].T, color=c, lw=1.8, alpha=0.85, zorder=3)
        ax.scatter(*P[start], s=90, color=NAVY, zorder=4, edgecolor="white")
        h20 = r[(r.q == q) & (r.t == 20)].hop.iloc[0]
        ax.set_title(f"q = {q}  →  20걸음 뒤 평균 {h20:.1f} hop", color=NAVY)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = axes[2]
    cols = {0.25: PURPLE, 0.5: BLUE, 1: GREY, 2: OLIVE, 4: TEAL}
    for q, c in cols.items():
        s_ = r[r.q == q]
        ax.plot(s_.t, s_.hop, color=c, lw=2.3, marker="o", ms=3.5, label=f"q = {q}")
    ax.set_xlabel("걸음 수 t"); ax.set_ylabel("출발점에서의 최단거리 (hop, 평균)")
    ax.set_title("q가 작을수록 멀리 간다 (p = 1)", color=NAVY)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "walks.png"))
    print(r[r.t.isin([5, 10, 20])].pivot(index="t", columns="q", values="hop").round(2))

    # ── 그림 2: 같은 뿌리 + p, q 재현 ────────────────────────────────
    d2, d3 = pd.read_csv(f2), pd.read_csv(f3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    M = [("Leiden", NAVY, "-", "o"), ("스펙트럴", ORANGE, "-", "s"), ("NetMF", RED, "-", "D"),
         ("DeepWalk", TEAL, "-", "o"), ("node2vec q=0.5", PURPLE, "--", "^"), ("node2vec q=2", OLIVE, "--", "v")]
    g = d2.groupby(["method", "p_out"]).ari.agg(["mean", "std"]).reset_index().fillna(0)
    for k, (m, c, ls, mk) in enumerate(M):
        s_ = g[g.method == m]
        ax.errorbar(s_.p_out + (k - 2.5) * 0.0004, s_["mean"], yerr=s_["std"], color=c, ls=ls,
                    marker=mk, lw=2, ms=6, capsize=3, label=m)
    ax.set_xticks(P_OUTS); ax.set_xticklabels(["0.010\n(쉬움)", "0.018", "0.026\n(어려움)"])
    ax.set_xlabel("커뮤니티 사이 연결 확률"); ax.set_ylabel("정답 커뮤니티와의 ARI")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("05 §2의 그래프 — 정답 커뮤니티를 얼마나 찾나", color=NAVY)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", ncol=2)
    print(g.pivot(index="method", columns="p_out", values="mean").round(3))
    ax = axes[1]
    for j, (col, lab) in enumerate([("ari_comm", "Leiden 커뮤니티와의 ARI"), ("eta2_deg", "군집이 설명하는\n차수 분산 (η²)")]):
        for k, (q, c) in enumerate([(0.5, PURPLE), (2, OLIVE)]):
            v = d3[d3.q == q][col].values
            x = j + (k - 0.5) * 0.35
            ax.bar(x, v.mean(), 0.32, color=c, alpha=0.85, label=f"q = {q}" if j == 0 else None)
            ax.scatter(np.full(len(v), x) + np.linspace(-0.08, 0.08, len(v)), v, s=12, color=NAVY, zorder=3)
            ax.text(x, v.max() + 0.03, f"{v.mean():.2f}", ha="center", fontsize=10.5, color=NAVY)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Leiden 커뮤니티와의 ARI", "군집이 설명하는 차수 분산 (η²)"])
    ax.set_ylim(0, 1); ax.set_ylabel("값 (시드 10개)")
    ax.set_title("Les Misérables — 논문 설정 재현 (p = 1)", color=NAVY)
    ax.legend(frameon=False, loc="upper right")
    print(d3.groupby("q")[["ari_comm", "eta2_deg"]].agg(["mean", "std"]).round(3))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "compare.png"))
    print("done")

"""04 §2 — null model 과 Z-score.

관측값 하나만으로는 아무 말도 할 수 없다. 차수를 보존한 채 엣지를 섞어
같은 지표를 1000번 다시 재면, 관측값이 어디쯤인지 비로소 말할 수 있다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx

OUT = outdir("c04")

G = nx.karate_club_graph()
m = G.number_of_edges()
obs_C = nx.average_clustering(G)
obs_T = sum(nx.triangles(G).values()) // 3

rng = np.random.default_rng(0)
N = 1000
null_C, null_T = [], []
for i in range(N):
    H = G.copy()
    # 차수를 그대로 둔 채 엣지 양끝을 맞바꾸는 연산 (degree-preserving rewiring)
    nx.double_edge_swap(H, nswap=10 * m, max_tries=200 * m, seed=int(rng.integers(1e9)))
    null_C.append(nx.average_clustering(H))
    null_T.append(sum(nx.triangles(H).values()) // 3)
null_C, null_T = np.array(null_C), np.array(null_T)

def zscore(obs, null):
    return (obs - null.mean()) / null.std(ddof=1)

def pval(obs, null):
    # 단측 경험적 p-value (관측값 이상이 나온 비율), +1 보정
    return (np.sum(null >= obs) + 1) / (len(null) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), gridspec_kw={"wspace": 0.22})

for ax, (obs, null, label) in zip(axes, [
        (obs_C, null_C, "평균 뭉침 계수  C"),
        (obs_T, null_T, "삼각형 수")]):
    ax.hist(null, bins=28, color=GREY, alpha=0.85,
            label=f"null 분포 (재배선 {N}회)")
    ax.axvline(obs, color=RED, lw=2.6, label="관측값")
    ax.axvline(null.mean(), color=NAVY, lw=1.4, ls="--", label="null 평균")
    z, p = zscore(obs, null), pval(obs, null)
    ax.set_xlabel(label); ax.set_ylabel("빈도")
    # 재배선 N회로 낼 수 있는 가장 작은 p 는 1/(N+1) 입니다
    ptxt = f"p ≤ {1/(N+1):.3f}" if p <= 1 / (N + 1) else f"p = {p:.3f}"
    ax.set_title(f"{label}   —   Z = {z:.1f},  {ptxt}", color=NAVY,
                 fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    print(f"{label}: 관측 {obs if isinstance(obs,int) else round(obs,4)} | "
          f"null 평균 {null.mean():.4f} ± {null.std(ddof=1):.4f} | "
          f"Z = {z:.2f} | p = {p:.4f}")

fig.suptitle("관측값(빨강)이 null 분포(회색)에서 얼마나 떨어져 있는가",
             color=NAVY, fontsize=13, fontweight="bold", y=1.02)
fig.savefig(os.path.join(OUT, "nullmodel.png"))
print("wrote", os.path.join(OUT, "nullmodel.png"))

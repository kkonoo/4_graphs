"""07 §2 — 연구 편향(study bias)이 만드는 가짜 허브.

진짜 네트워크에는 허브가 하나도 없다(ER). 그런데 '많이 연구된 단백질일수록
상호작용이 발견될 확률이 높다'는 규칙 하나만 넣으면, 관측된 네트워크는
꼬리가 두꺼워지고 허브가 생긴 것처럼 보인다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import networkx as nx
from scipy.stats import spearmanr

OUT = outdir("c07")
rng = np.random.default_rng(0)

n, m = 1000, 6000
G_true = nx.gnm_random_graph(n, m, seed=1)
deg_true = np.array([G_true.degree(v) for v in range(n)])

# 연구량: 로그정규 — 소수의 유전자에 연구가 몰려 있다
effort = rng.lognormal(0, 1.2, n)
effort = np.clip(effort / np.percentile(effort, 99), 0.02, 1.0)

# 엣지 발견 확률은 양쪽 유전자의 연구량에 달려 있다
G_obs = nx.Graph(); G_obs.add_nodes_from(range(n))
for u, v in G_true.edges():
    if rng.random() < (effort[u] * effort[v]) ** 0.35:
        G_obs.add_edge(u, v)
deg_obs = np.array([G_obs.degree(v) for v in range(n)])

rho_true = spearmanr(deg_obs, deg_true).statistic
rho_eff = spearmanr(deg_obs, effort).statistic

fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.5), gridspec_kw={"wspace": 0.28})

# --- A. 차수 분포 (평균으로 나눠 같은 자에 놓고 비교) ----------------------
ax = axes[0]
bins = np.linspace(0, 5, 34)
ax.hist(deg_true / deg_true.mean(), bins=bins, color=BLUE, alpha=0.8,
        label=f"진짜 (ER)  CV={deg_true.std()/deg_true.mean():.2f}")
ax.hist(deg_obs / deg_obs.mean(), bins=bins, color=RED, alpha=0.7,
        label=f"관측  CV={deg_obs.std()/deg_obs.mean():.2f}")
ax.set_xlabel("차수 / 평균 차수"); ax.set_ylabel("노드 수")
ax.set_title("없던 꼬리가 생긴다", color=NAVY, fontsize=12)
ax.legend(frameon=False, fontsize=10)

# --- B. 관측 차수는 무엇을 따라가는가 --------------------------------------
ax = axes[1]
# 차수가 정수라 점이 줄무늬로 겹칩니다 — 세로로 살짝 흔들어 표시합니다
jit = rng.normal(0, 0.14, n)
ax.scatter(effort, deg_obs + jit, s=8, color=RED, alpha=0.30)
qs = np.quantile(effort, np.linspace(0, 1, 11))
cx, cy = [], []
for a, b in zip(qs[:-1], qs[1:]):
    sel = (effort >= a) & (effort < b)
    if sel.sum():
        cx.append(effort[sel].mean()); cy.append(deg_obs[sel].mean())
ax.plot(cx, cy, "o-", color=NAVY, lw=2.2, ms=6, label="구간 평균")
ax.legend(frameon=False, fontsize=10)
ax.set_xlabel("연구량 (상대값)"); ax.set_ylabel("관측된 차수")
ax.set_title(f"관측 차수 vs 연구량   ρ = {rho_eff:.2f}", color=NAVY, fontsize=12)

# --- C. 상관 비교 ----------------------------------------------------------
ax = axes[2]
ax.bar(["진짜 차수", "연구량"], [rho_true, rho_eff], color=[BLUE, RED], width=0.5)
for i, val in enumerate([rho_true, rho_eff]):
    ax.text(i, val + 0.015, f"{val:.2f}", ha="center", fontsize=12,
            fontweight="bold", color=[BLUE, RED][i])
ax.set_ylim(0, 0.6); ax.set_ylabel("관측 차수와의 스피어만 상관 ρ")
ax.set_title("⚠ 관측 허브는 진짜 차수보다\n연구량을 더 잘 따라간다", color=NAVY,
             fontsize=12)

fig.suptitle("진짜 네트워크에는 허브가 없다 — 허브는 관측 과정이 만들었다",
             color=NAVY, fontsize=13, fontweight="bold", y=1.03)
fig.savefig(os.path.join(OUT, "study_bias.png"))
print("wrote", os.path.join(OUT, "study_bias.png"))
print(f"진짜: 평균 {deg_true.mean():.2f} CV {deg_true.std()/deg_true.mean():.3f} "
      f"최대/평균 {deg_true.max()/deg_true.mean():.2f}")
print(f"관측: 평균 {deg_obs.mean():.2f} CV {deg_obs.std()/deg_obs.mean():.3f} "
      f"최대/평균 {deg_obs.max()/deg_obs.mean():.2f} | 차수 0인 노드 {(deg_obs==0).sum()}개")
print(f"관측 엣지 {G_obs.number_of_edges()} / {m} ({G_obs.number_of_edges()/m:.0%})")
print(f"rho(진짜차수) {rho_true:.3f} | rho(연구량) {rho_eff:.3f}")
t = set(np.argsort(deg_obs)[::-1][:50])
print(f"관측 상위50 ∩ 진짜 상위50 = {len(t & set(np.argsort(deg_true)[::-1][:50]))}개 | "
      f"∩ 연구량 상위50 = {len(t & set(np.argsort(effort)[::-1][:50]))}개")

"""07 §2-1 — 연구 편향(study bias)이 만드는 가짜 허브.

진짜 네트워크에는 허브가 하나도 없다(ER). 그런데 '많이 연구된 단백질일수록
상호작용이 발견될 확률이 높다'는 규칙 하나만 넣으면, 관측된 네트워크는
꼬리가 두꺼워지고 허브가 생긴 것처럼 보인다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medfig import *
import networkx as nx
from scipy.stats import spearmanr

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

cv_true = deg_true.std() / deg_true.mean()
cv_obs = deg_obs.std() / deg_obs.mean()

# ── 색: 진짜 네트워크 · 관측 네트워크 · 연구량 (그림 안에서 같은 뜻 = 같은 색) ──
C_TRUE, C_OBS, C_EFF = CAT[1], PRIMARY, CAT[2]

plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
fig = plt.figure(figsize=(8.4, 6.3))
gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], height_ratios=[1, 1.05],
                      hspace=0.55, wspace=0.42)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

def marker(ax, text):
    ax.text(0.0, 1.06, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12.5, fontweight="bold", color=TEXT)

# --- ① 차수 분포 (평균으로 나눠 같은 자에 놓고 비교) ----------------------
ax = ax1
bins = np.linspace(0, 5, 34)
ax.hist(deg_obs / deg_obs.mean(), bins=bins, color=C_OBS, lw=0, zorder=2)
# 진짜 분포는 윤곽선으로 — 값이 있는 구간만 그려 축 위를 덮지 않게
h_true, _ = np.histogram(deg_true / deg_true.mean(), bins=bins)
nz = np.nonzero(h_true)[0]
lo, hi = nz.min(), nz.max() + 1
ax.stairs(h_true[lo:hi], bins[lo:hi + 1], baseline=0, fill=False, color=C_TRUE, lw=2.2, zorder=3)
ax.set_xlabel("차수 / 평균 차수", fontsize=11.5)
ax.set_ylabel("노드 수", fontsize=11.5)
ax.set_xlim(-0.1, 5.05); ax.set_ylim(0, 290)
ax.text(2.25, 240, f"진짜 (ER)   CV = {cv_true:.2f}", color=C_TRUE, fontsize=11.5,
        fontweight="bold", va="center")
ax.text(2.25, 205, f"관측   CV = {cv_obs:.2f}", color=C_OBS, fontsize=11.5,
        fontweight="bold", va="center")
ax.annotate(f"없던 꼬리 · 최대 차수가 평균의 {deg_obs.max() / deg_obs.mean():.2f}배",
            xy=(2.75, 38), xytext=(3.0, 120), fontsize=11, color=ACCENT_INK,
            fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-|>", color=ACCENT_INK, lw=1.2,
                            connectionstyle="arc3,rad=0.2"))
marker(ax, "① 차수 분포")

# --- ② 관측 차수는 무엇을 따라가는가 --------------------------------------
ax = ax2
# 차수가 정수라 점이 줄무늬로 겹칩니다 — 세로로 살짝 흔들어 표시합니다
jit = rng.normal(0, 0.14, n)
ax.scatter(effort, deg_obs + jit, s=8, color=mix(C_OBS, 0.55), alpha=0.45, lw=0, zorder=2)
qs = np.quantile(effort, np.linspace(0, 1, 11))
cx, cy = [], []
for a, b in zip(qs[:-1], qs[1:]):
    sel = (effort >= a) & (effort < b)
    if sel.sum():
        cx.append(effort[sel].mean()); cy.append(deg_obs[sel].mean())
MEAN_C = shade(C_OBS, 0.35)
ax.plot(cx, cy, "o-", color=MEAN_C, lw=2.2, ms=5.5, zorder=4)
ax.text(cx[-1] + 0.04, cy[-1], "구간 평균", color=MEAN_C, fontsize=11, fontweight="bold",
        va="center", ha="left", zorder=5,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
ax.text(0.97, 0.97, f"ρ = {rho_eff:.2f}", transform=ax.transAxes, ha="right", va="top",
        fontsize=12, fontweight="bold", color=TEXT)
ax.set_xlabel("연구량 (상대값)", fontsize=11.5)
ax.set_ylabel("관측된 차수", fontsize=11.5)
ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.8, 10.8)
ax.set_yticks([0, 2, 4, 6, 8, 10])
marker(ax, "② 관측 차수 vs 연구량")

# --- ③ 상관 비교 ----------------------------------------------------------
ax = ax3
cols = [C_TRUE, C_EFF]
ax.bar([0, 1], [rho_true, rho_eff], color=cols, width=0.55, zorder=2)
for i, val in enumerate([rho_true, rho_eff]):
    ax.text(i, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=12,
            fontweight="bold", color=cols[i])
ax.set_xticks([0, 1]); ax.set_xticklabels(["진짜 차수", "연구량"])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0, 0.6); ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.set_ylabel("관측 차수와의\n스피어만 상관 ρ", fontsize=11.5)
marker(ax, "③ 무엇을 더 따라가나")

save(fig, img("c07", "study_bias.png"))
print(f"진짜: 평균 {deg_true.mean():.2f} CV {deg_true.std()/deg_true.mean():.3f} "
      f"최대/평균 {deg_true.max()/deg_true.mean():.2f}")
print(f"관측: 평균 {deg_obs.mean():.2f} CV {deg_obs.std()/deg_obs.mean():.3f} "
      f"최대/평균 {deg_obs.max()/deg_obs.mean():.2f} | 차수 0인 노드 {(deg_obs==0).sum()}개")
print(f"관측 엣지 {G_obs.number_of_edges()} / {m} ({G_obs.number_of_edges()/m:.0%})")
print(f"rho(진짜차수) {rho_true:.3f} | rho(연구량) {rho_eff:.3f}")
t = set(np.argsort(deg_obs)[::-1][:50])
print(f"관측 상위50 ∩ 진짜 상위50 = {len(t & set(np.argsort(deg_true)[::-1][:50]))}개 | "
      f"∩ 연구량 상위50 = {len(t & set(np.argsort(effort)[::-1][:50]))}개")

"""10·13 공용 — PBMC 3k (scanpy 튜토리얼의 전처리 완료본) 불러오기.

sc.datasets.pbmc3k_processed(): 세포 2,638개 x 고변동 유전자 1,838개 (정규화·로그·
회귀·표준화까지 끝난 값), 세포유형 라벨 8개(obs['louvain']).
figs/cache/pbmc3k.npz 에 저장해두고 재사용 (원자료 캐시는 커밋하지 않음).
"""
import os
import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load_pbmc():
    f = os.path.join(CACHE, "pbmc3k.npz")
    if not os.path.exists(f):
        import scanpy as sc
        sc.settings.datasetdir = CACHE
        a = sc.datasets.pbmc3k_processed()
        np.savez(f, X=np.asarray(a.X, dtype=np.float32),
                 labels=np.array(a.obs["louvain"].astype(str).tolist(), dtype=str),
                 cells=np.array(a.obs_names, dtype=str), genes=np.array(a.var_names, dtype=str),
                 pca=np.asarray(a.obsm["X_pca"], dtype=np.float32))
    z = np.load(f, allow_pickle=False)
    return z["X"], z["labels"], z["pca"]

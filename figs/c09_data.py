"""09 공용 — TCGA BRCA PanCanAtlas 발현 + PAM50 아형 불러오기.

cBioPortal datahub(GitHub LFS)에서 받아 figs/cache/ 에 저장해두고 재사용합니다.
본문 🐍 실습과 같은 전처리: log2(RSEM + 1), 환자 ID(앞 12자리)로 임상과 연결.
"""
import os
import numpy as np
import pandas as pd

BASE = ("https://media.githubusercontent.com/media/cBioPortal/datahub/master/"
        "public/brca_tcga_pan_can_atlas_2018/")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load_brca():
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, "brca.npz")                  # pandas 버전과 무관한 형식으로 캐시
    if os.path.exists(f):
        z = np.load(f, allow_pickle=False)
        X = pd.DataFrame(z["X"], index=z["patients"], columns=z["genes"])
        return X, pd.Series(z["subtype"], index=z["patients"], name="SUBTYPE")
    expr = pd.read_csv(BASE + "data_mrna_seq_v2_rsem.txt", sep="\t")
    clin = pd.read_csv(BASE + "data_clinical_patient.txt", sep="\t", comment="#")
    expr = (expr.dropna(subset=["Hugo_Symbol"]).drop_duplicates("Hugo_Symbol")
                .set_index("Hugo_Symbol").drop(columns="Entrez_Gene_Id"))
    X = np.log2(expr.T.astype(float) + 1)
    X.index = X.index.str[:12]
    X = X[~X.index.duplicated()]
    lab = clin.set_index("PATIENT_ID")["SUBTYPE"].dropna()
    common = X.index.intersection(lab.index)
    X, y = X.loc[common], lab.loc[common].str.replace("BRCA_", "")
    np.savez(f, X=X.values, patients=np.array(X.index, dtype=str),
             genes=np.array(X.columns, dtype=str), subtype=np.array(y.values, dtype=str))
    return load_brca()

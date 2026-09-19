"""강의 그림 공통 스타일.

모든 그림 스크립트는 `from style import *` 로 시작하고 OUT 아래에 저장합니다.
한글 폰트는 실행 환경(Windows / macOS / Linux)에 따라 자동으로 고릅니다.
"""
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

# ── 팔레트 ────────────────────────────────────────────────────────────
TEAL = "#5F9EA0"; PURPLE = "#9B8EC4"; OLIVE = "#B5AE8A"; NAVY = "#2E4057"
RED = "#C1544B"; GREY = "#B8B8B8"; LGREY = "#E8E8E8"; ORANGE = "#E08A3C"
BLUE = "#4A77A8"
# 사이트 CSS의 제목 색과 맞춘 보조색
PLUM = "#5B4D80"


# ── 한글 폰트 ─────────────────────────────────────────────────────────
def _korean_font():
    """설치된 한글 폰트 이름을 반환. 없으면 None."""
    for name in ("Pretendard", "Malgun Gothic", "AppleGothic",
                 "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        try:
            if fm.findfont(fm.FontProperties(family=name),
                           fallback_to_default=False):
                return name
        except Exception:
            pass
    # 리눅스: 파일 경로로 직접 등록 (Noto CJK 는 KR/JP 가 한 파일에 들어 있음)
    for pat in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/**/NanumGothic*.ttf",
                "/usr/share/fonts/**/NotoSansCJK*.ttc"):
        for path in glob.glob(pat, recursive=True):
            try:
                fm.fontManager.addfont(path)
                return fm.FontProperties(fname=path).get_name()
            except Exception:
                continue
    return None


KO = _korean_font()

# 한글 폰트를 앞에, DejaVu Sans 를 뒤에 두면 한글 폰트에 없는 글자
# (예: Erdős 의 ő)는 DejaVu 로 대체됩니다.
plt.rcParams.update({
    "font.family": ([KO, "DejaVu Sans"] if KO else ["DejaVu Sans"]),
    "font.size": 11,
    "axes.unicode_minus": False,        # 한글 폰트에서 마이너스 깨짐 방지
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4A4A4A", "axes.labelcolor": "#2A2A2A",
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlepad": 12,
    "xtick.color": "#4A4A4A", "ytick.color": "#4A4A4A",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})

# 각 스크립트에서 chapter 폴더를 붙여 씁니다.
IMG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")


def outdir(chapter):
    """images/<chapter>/ 를 만들고 경로를 돌려줍니다."""
    d = os.path.join(IMG_ROOT, chapter)
    os.makedirs(d, exist_ok=True)
    return d

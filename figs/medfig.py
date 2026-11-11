"""Shared look for course concept figures — HaLab slide-gallery palette, gradient mode.

This repo: figs/medfig.py, THEME = violet (assets/html.css --course-accent #4B2C7F).
One script per figure, figs/cNN_name.py:  from medfig import *  →  save(fig, img("cNN", "name.png")).

Rules this module encodes (see the skill's SKILL.md for the why):
- Fills are flat. "Gradient mode" means using more hues between the theme colour and its
  gradient partner, each as a SOLID colour (CAT, ramp(n)) — never a gradient inside a box.
- Parallel items (types, steps, groups) take CAT / ramp colours, not the accent.
- ACCENT marks at most one thing per figure, and only a single must-see point. Often none.
- Korean labels in Pretendard; no titles; ≤ 8.5 in wide; text ≥ 11 pt.

Coordinates for drawings are inches: new(w, h) gives an axis running 0..w and 0..h.
"""
import glob, logging, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.transforms import Bbox
import numpy as np

THEME = os.environ.get("FIG_THEME", "violet")    # ← site subject colour (assets/html.css)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
for f in glob.glob(os.path.expanduser("~/.fonts/Pretendard-*.otf")):
    fm.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Pretendard", "DejaVu Sans"], "font.size": 11,
    "text.color": "#262B33", "axes.labelcolor": "#262B33",
    "axes.edgecolor": "#9AA1AB", "xtick.color": "#4B5563", "ytick.color": "#4B5563",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.unicode_minus": False,
})

# ── palettes (kkonoo.github.io/slide-gallery → 색 코드) ─────────────────────────
# primary, steps (35·60·82%·옅은 면), gradient stops, accent in gradient mode, text on accent.
# cat = parallel-item colours (primary, gradient partners, extras). ramp = stops for ordered steps
# (defaults to the gradient). Non-crimson cat lists are first drafts: check preview_palette.py.
THEMES = {
    "crimson":  dict(primary="#B0283C", steps=["#CC7380", "#DFA9B1", "#F1D8DC", "#F7EAEC"],
                     grad=["#B0283C", "#C2531F"], accent="#2E9CCA", on_accent="#FFFFFF",
                     cat=["#B0283C", "#C2531F", "#7A1F3D", "#D9822B"],
                     ramp=["#7A1F3D", "#B0283C", "#C2531F", "#D9822B"]),
    "wine":     dict(primary="#7A1F3D", steps=["#A96D81", "#CAA5B1", "#E7D7DC", "#F2E9EC"],
                     grad=["#7A1F3D", "#B8323F", "#C2531F"], accent="#8FCBEA", on_accent="#431122",
                     cat=["#7A1F3D", "#B8323F", "#C2531F", "#D9822B"],
                     ramp=["#7A1F3D", "#B8323F", "#C2531F", "#D9822B"]),
    "teal":     dict(primary="#0F4C6E", steps=["#638BA1", "#9FB7C5", "#D4DFE5", "#E7EDF0"],
                     grad=["#0F4C6E", "#178270", "#2F7F3A"], accent="#FFD93D", on_accent="#0F4C6E",
                     cat=["#0F4C6E", "#178270", "#2F7F3A", "#5A9BC0"]),
    "blue":     dict(primary="#2563C9", steps=["#719ADC", "#A8C1E9", "#D8E3F5", "#E9EFFA"],
                     grad=["#2563C9", "#0E7C8A", "#0D7A5A"], accent="#FFD93D", on_accent="#0D2250",
                     cat=["#2563C9", "#0E7C8A", "#0D7A5A", "#7FA6E3"]),
    "forest":   dict(primary="#1F5236", steps=["#6D8F7C", "#A5BAAF", "#D7E0DB", "#E9EEEB"],
                     grad=["#1F5236", "#5E7F25"], accent="#F2B33D", on_accent="#112D1E",
                     cat=["#1F5236", "#5E7F25", "#3F8F6B", "#A3B85C"]),
    "emerald":  dict(primary="#0B6E5A", steps=["#60A194", "#9DC5BD", "#D3E5E1", "#E7F0EE"],
                     grad=["#0B6E5A", "#16708F"], accent="#7B4FE0", on_accent="#FFFFFF",
                     cat=["#0B6E5A", "#16708F", "#5FB3A0", "#6FA3C8"]),
    "violet":   dict(primary="#4B2C7F", steps=["#8A76AC", "#B7ABCC", "#DFD9E8", "#EDEAF2"],
                     grad=["#4B2C7F", "#2F5BC8", "#0C7E7E"], accent="#FF8A78", on_accent="#291846"),
    "plum":     dict(primary="#7A2E6B", steps=["#A9779F", "#CAABC4", "#E7D9E4", "#F2EAF0"],
                     grad=["#7A2E6B", "#C0405E"], accent="#2BB3A6", on_accent="#43193B"),
    "charcoal": dict(primary="#2F343B", steps=["#787B80", "#ACAEB1", "#DADADC", "#EAEBEB"],
                     grad=["#2F343B", "#33496B", "#3F6A92"], accent="#E5484D", on_accent="#FFFFFF"),
    "graphite": dict(primary="#4A4F57", steps=["#898D92", "#B7B9BC", "#DEDFE1", "#EDEDEE"],
                     grad=["#4A4F57", "#3F6A92"], accent="#FFD43B", on_accent="#292B30",
                     cat=["#4A4F57", "#3F6A92", "#8C9199", "#9DB6CF"]),
}

def mix(c, a):
    """Colour c at strength a (0–1) over white, as a solid colour."""
    c = np.array(to_rgb(c))
    return to_hex(1 + (c - 1) * a)

def shade(c, a):
    """Colour c darkened toward black by a (0–1)."""
    return to_hex(np.array(to_rgb(c)) * (1 - a))

def _lum(c):
    r, g, b = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in to_rgb(c)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _derive_cat(t):
    stops = [t["primary"]] + [g for g in t["grad"] if g.upper() != t["primary"].upper()]
    p = t["primary"]
    extra = [mix(p, 0.6)] if _lum(p) < 0.06 else [shade(p, 0.35)]   # very dark theme → lighter extra
    extra.append(mix(stops[-1], 0.6))
    return (stops + extra)[:4]

T = THEMES[THEME]
PRIMARY = T["primary"]
STEPS = T["steps"]                       # 35%, 60%, 82%, 옅은 면 — tints of the primary
CAT = T.get("cat") or _derive_cat(T)     # parallel items, in this order
ACCENT = T["accent"]                     # fill colour for the one key point
ON_ACCENT = T["on_accent"]               # text placed on an ACCENT fill
# Text or thin lines in the accent colour need contrast on white: darken light accents.
ACCENT_INK = ACCENT if _lum(ACCENT) < 0.3 else shade(ACCENT, 0.45)
TEXT, TEXT2, LINE, FAINT = "#262B33", "#6B7280", "#E2E5EA", "#F5F6F8"
GRAD = LinearSegmentedColormap.from_list("theme", T.get("ramp", T["grad"]))

IMG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")

def img(chapter, name):
    """images/<chapter>/<name> — where the chapters look for figures."""
    return os.path.join(IMG_ROOT, chapter, name)

def ramp(n):
    """n solid colours along the theme's gradient (for ordered steps 1..n)."""
    return [to_hex(GRAD(i / max(n - 1, 1))) for i in range(n)]

def on(c):
    """Readable text colour on fill c (white on dark fills, TEXT on light ones)."""
    return "white" if _lum(c) < 0.35 else TEXT

# ── canvas ──────────────────────────────────────────────────────────────────
def new(w, h):
    """Drawing canvas w × h inches with an axis in inches. Keep w ≤ 8.5."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax

def save(fig, path):
    """PNG at 200 dpi. A drawing-only figure (all axes off) is cropped to what is drawn."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.canvas.draw()
    bbox = "tight"
    w = fig.get_tightbbox(fig.canvas.get_renderer()).width + 0.24
    if not any(a.axison for a in fig.axes):
        r = fig.canvas.get_renderer()           # (drawn above: scatter extents need a draw)
        arts = [x for a in fig.axes for x in a.texts + a.patches + a.images + a.lines + a.collections
                if x.get_visible()]
        bbs = [x.get_window_extent(r) for x in arts]
        bb = Bbox.union([b for b in bbs if np.isfinite(b.bounds).all() and b.width > 0 and b.height > 0])
        bbox = bb.transformed(fig.dpi_scale_trans.inverted()).padded(0.12)
        w = bbox.width
    fig.savefig(path, dpi=200, bbox_inches=bbox, pad_inches=0.12)
    plt.close(fig)
    print(f"saved {path}  (~{w:.1f} in wide)" + ("  ⚠ wider than 8.5 in" if w > 8.6 else ""))

# ── shapes ──────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, fc="white", ec=LINE, lw=1.0, r=0.08, z=2, ls="-"):
    """Rounded box with its lower-left corner at (x, y). Flat fill only."""
    p = mp.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                          fc=fc, ec=ec, lw=lw, zorder=z, ls=ls)
    ax.add_patch(p)
    return p

def chevron(ax, x, y, w, h, fc, tip=0.18, first=False, z=2):
    """Arrow-shaped step box (flat left edge if first)."""
    pts = [(x, y), (x + w - tip, y), (x + w, y + h / 2), (x + w - tip, y + h), (x, y + h)]
    if not first:
        pts.append((x + tip, y + h / 2))
    ax.add_patch(mp.Polygon(pts, closed=True, fc=fc, ec="none", zorder=z))

def num(ax, x, y, n, fc=None, tc=None, r=0.15, fs=11, z=4):
    """Numbered circle centred at (x, y)."""
    fc = fc or PRIMARY
    ax.add_patch(mp.Circle((x, y), r, fc=fc, ec="none", zorder=z))
    ax.text(x, y, str(n), ha="center", va="center", color=tc or on(fc), fontsize=fs,
            fontweight="bold", zorder=z + 1)

def pill(ax, x, y, text, fc=None, tc=None, fs=11, padx=0.12, h=0.3, z=4, ha="center"):
    """Text in a rounded label; (x, y) is the centre (or left edge if ha='left')."""
    fc = fc or PRIMARY
    t = ax.text(0, 0, text, fontsize=fs, fontweight="bold")
    ax.figure.canvas.draw()
    w = t.get_window_extent().width / ax.figure.dpi + 2 * padx
    t.remove()
    x0 = x - w / 2 if ha == "center" else x
    box(ax, x0, y - h / 2, w, h, fc=fc, ec="none", r=h / 2, z=z)
    ax.text(x0 + w / 2, y, text, ha="center", va="center", fontsize=fs, fontweight="bold",
            color=tc or on(fc), zorder=z + 1)
    return w

def arrow(ax, x1, y1, x2, y2, color="#9AA1AB", lw=1.6, ms=12, z=3, ls="-", rad=0):
    ax.add_patch(mp.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                    color=color, lw=lw, zorder=z, ls=ls,
                                    connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

def label(ax, x, y, text, fs=11, w="regular", c=TEXT, ha="center", va="center", z=5, **kw):
    return ax.text(x, y, text, fontsize=fs, fontweight=w, color=c, ha=ha, va=va, zorder=z, **kw)

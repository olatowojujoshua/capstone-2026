import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Design Tokens ─────────────────────────────────────────────────────
BG_COLOR      = "#0f0f1a"
PANEL_COLOR   = "#13132b"
GRID_COLOR    = "#1e1e3a"
TEXT_COLOR    = "#e0e0f0"
SUBTEXT_COLOR = "#8888aa"
ACCENT_COLORS = ["#00d4ff", "#7c4dff", "#ff4081", "#ffab40", "#69f0ae", "#40c4ff"]
FONT_FAMILY   = "Segoe UI"

# Gradient colormaps
CYAN_PURPLE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "cyan_purple", ["#00d4ff", "#7c4dff"]
)
NEON_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "neon", ["#00d4ff", "#7c4dff", "#ff4081"]
)


def apply_dark_style(ax, fig, title="", xlabel="", ylabel="", title_size=18):
    """Apply the shared premium dark theme to any axes."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(0.6)

    # Grid
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Title & labels
    ax.set_title(
        title, fontsize=title_size, fontweight="bold",
        color=TEXT_COLOR, fontfamily=FONT_FAMILY, pad=18
    )
    ax.set_xlabel(
        xlabel, fontsize=11, color=SUBTEXT_COLOR,
        fontfamily=FONT_FAMILY, labelpad=10
    )
    ax.set_ylabel(
        ylabel, fontsize=11, color=SUBTEXT_COLOR,
        fontfamily=FONT_FAMILY, labelpad=10
    )

    # Tick styling
    ax.tick_params(colors=SUBTEXT_COLOR, labelsize=10, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)


def add_glow_line(ax, x, y, color="#00d4ff", linewidth=2.5, label=None):
    """Draw a line with a neon glow effect (layered alpha lines)."""
    for width, alpha in [(linewidth * 4, 0.06), (linewidth * 2.5, 0.12), (linewidth * 1.5, 0.25)]:
        ax.plot(x, y, color=color, linewidth=width, alpha=alpha, solid_capstyle="round")
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=1.0,
            solid_capstyle="round", label=label)


def add_gradient_fill(ax, x, y, color_bottom="#00d4ff", color_top="#7c4dff",
                      alpha=0.3, y_min=None):
    """Add a vertical gradient fill under a curve using imshow + clip."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    if y_min is None:
        y_min = ax.get_ylim()[0]

    # Build closed polygon
    verts = list(zip(x, y)) + [(x[-1], y_min), (x[0], y_min)]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    path = MplPath(verts, codes)
    patch = PathPatch(path, facecolor="none", edgecolor="none")
    ax.add_patch(patch)

    # Gradient image
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = mcolors.LinearSegmentedColormap.from_list("gf", [color_bottom, color_top])
    ax.imshow(
        gradient, aspect="auto", origin="lower",
        extent=[min(x), max(x), y_min, max(y) * 1.01],
        cmap=cmap, alpha=alpha, clip_path=patch, clip_on=True
    )


def save_fig(name):
    """Save with tight layout, high DPI, dark background."""
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"{name}.png",
        dpi=200,
        facecolor=BG_COLOR,
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.close()
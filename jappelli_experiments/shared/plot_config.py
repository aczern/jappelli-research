"""
Consistent matplotlib/seaborn styling for all figures.

Usage:
    from jappelli_experiments.shared.plot_config import setup_plots, save_fig
    setup_plots()
    fig, ax = plt.subplots()
    ...
    save_fig(fig, 'a1_martingale_test.pdf')
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from jappelli_experiments.config import FIGURE_DIR

# ── Color palette ──
COLORS = {
    "primary": "#2C3E50",
    "secondary": "#E74C3C",
    "tertiary": "#3498DB",
    "quaternary": "#2ECC71",
    "quinary": "#F39C12",
    "gray": "#95A5A6",
    "light_gray": "#BDC3C7",
    "recession": "#D5D8DC",
}

PALETTE = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
           COLORS["quaternary"], COLORS["quinary"]]


def setup_plots():
    """Apply consistent plot styling."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })
    sns.set_palette(PALETTE)


def save_fig(fig, filename, tight=True):
    """
    Save figure to the output/figures/ directory.

    Parameters
    ----------
    fig : matplotlib Figure
    filename : str
        Output filename (e.g., 'a1_martingale_test.pdf').
    """
    path = FIGURE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight" if tight else None)
    plt.close(fig)
    return path


def add_recession_bars(ax, recessions):
    """
    Add NBER recession shading to a time-series plot.

    Parameters
    ----------
    ax : matplotlib Axes
    recessions : DataFrame
        With columns 'start' and 'end' (datetime).
    """
    for _, row in recessions.iterrows():
        ax.axvspan(row["start"], row["end"],
                    color=COLORS["recession"], alpha=0.3, zorder=0)


def format_date_axis(ax, freq="yearly"):
    """Format x-axis for time series with date labels."""
    if freq == "yearly":
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    elif freq == "quarterly":
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    elif freq == "monthly":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

"""Shared chart style so notebooks and the app read as one system.

Colors follow the entity (a model keeps its color everywhere), one y-axis
per chart, thin marks, recessive grid.
"""
from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8986"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

MODEL_COLORS = {
    # demand
    "Seasonal naive": NEUTRAL,
    "HDD regression": SERIES[0],
    "SARIMAX": SERIES[1],
    "Prophet": SERIES[2],
    "LightGBM": SERIES[3],
    "Hybrid (HDD reg + LightGBM)": SERIES[4],
    # price
    "Random walk": NEUTRAL,
    "Ridge (momentum only)": SERIES[0],
    "Ridge (fundamentals)": SERIES[1],
    "Ridge (all features)": SERIES[2],
    "LightGBM (all features)": SERIES[3],
}

SECTOR_COLORS = {
    "residential": SERIES[0], "commercial": SERIES[1],
    "industrial": SERIES[2], "electric_power": SERIES[3],
}

SCENARIO_COLORS = {"normal": SERIES[0], "cold": SERIES[6], "warm": SERIES[1]}


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.dpi": 110,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "lines.linewidth": 2, "legend.frameon": False,
        "font.size": 10, "figure.figsize": (10, 4.2),
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def source_note(ax, text: str) -> None:
    ax.annotate(text, xy=(0, -0.18), xycoords="axes fraction",
                fontsize=8, color=INK_2, ha="left", va="top")

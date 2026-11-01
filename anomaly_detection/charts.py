"""Charts for the anomaly detection results. Run after detect.py:
    python anomaly_detection/charts.py
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
MEASURED = "#2a78d6"      # blue
EXPECTED = "#8a8984"      # neutral gray
ANOMALY = "#e34948"       # red
RULE = "#eb6834"          # orange

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def shade_flags(ax, d, col, color, alpha):
    flag = d[col].astype(int)
    starts = d.loc[flag.diff().fillna(flag) == 1, "timestamp"]
    ends = d.loc[flag.diff().shift(-1).fillna(-flag) == -1, "timestamp"]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color=color, alpha=alpha, lw=0)


def event_chart(df: pd.DataFrame) -> None:
    """Dec 21: the largest anomaly, invisible to the fixed rule."""
    d = df[(df.timestamp >= "2024-12-21 06:00") & (df.timestamp < "2024-12-22 00:00")].copy()
    sm = d.set_index("timestamp")[["dev_mV", "expected_mV"]].rolling("3min").median().reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
    shade_flags(ax1, d, "ml_anomaly", ANOMALY, 0.12)
    ax1.plot(sm.timestamp, sm.expected_mV, color=EXPECTED, lw=2, ls="--", label="Expected for operating conditions (ML)")
    ax1.plot(sm.timestamp, sm.dev_mV, color=MEASURED, lw=2, label="Measured cell voltage spread")
    ax1.axhline(50, color=RULE, lw=1.2, ls=":")
    ax1.text(pd.Timestamp("2024-12-21 21:00", tz="UTC"), 51.5, "Fixed rule: 50 mV", color=INK2, fontsize=10)
    ax1.set_ylabel("Cell voltage spread (mV)")
    ax1.legend(loc="lower right", frameon=False, fontsize=10.5)

    peak = d.loc[d.residual_mV.idxmax()]
    ax1.annotate(
        f"{peak.dev_mV:.0f} mV measured\nvs {peak.expected_mV:.0f} mV expected\n"
        f"at {peak.SOC:.1f} % SOC → rule silent,\nML flags it\n"
        f"(similar discharges on\nDec 26/27 peaked ~55 mV)",
        xy=(peak.timestamp, peak.dev_mV), xytext=(pd.Timestamp("2024-12-21 20:10", tz="UTC"), 95),
        textcoords="data", va="top",
        fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=1),
    )

    ax2.plot(d.timestamp, d.SOC, color=INK2, lw=2)
    ax2.axhline(10, color=RULE, lw=1.2, ls=":")
    ax2.set_ylabel("SOC (%)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.set_xlabel("21 Dec 2024 (UTC)")

    fig.suptitle("Context-aware anomaly detection catches what a fixed threshold misses",
                 x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.915, "Red shading = ML anomaly (spread ≥ 10 mV above expected). "
             "Model scores each day using only the other days.", color=INK2, fontsize=10.5)
    fig.savefig(OUT / "chart_event_dec21.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def daily_chart(daily: pd.DataFrame) -> None:
    """Median excess spread per day: Dec 9-10 stand out as a whole period."""
    d = daily.reset_index()
    d["label"] = pd.to_datetime(d["day"]).dt.strftime("%d %b")
    colors = [ANOMALY if v >= 10 else EXPECTED for v in d.median_excess_mV]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(d.label, d.median_excess_mV, color=colors, width=0.7, edgecolor=SURFACE, linewidth=2)
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(10, color=ANOMALY, lw=1, ls=":")
    ax.text(len(d) - 0.5, 10.6, "10 mV anomaly threshold", ha="right", color=INK2, fontsize=10)
    for i, v in enumerate(d.median_excess_mV):
        if v >= 10:
            ax.text(i, v + 0.6, f"+{v:.0f}", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylabel("Median spread above expected (mV)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Dec 9–10: cell spread ran 11–15 mV above expected, then recovered",
                 x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.9, "Rule-based warnings on Dec 9: 0 (SOC stayed below 10 %).",
             color=INK2, fontsize=10.5)
    fig.savefig(OUT / "chart_daily_excess.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    scored = pd.read_parquet(OUT / "scored.parquet")
    daily = pd.read_csv(OUT / "daily_summary.csv")
    event_chart(scored)
    daily_chart(daily)
    print("Saved chart_event_dec21.png and chart_daily_excess.png")

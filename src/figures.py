"""Generate manuscript figures."""

import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import OUTPUT_TABLES, OUTPUT_FIGURES

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def figure1_wave_replication():
    """Coefficient plot: PGS cognition vs Alzheimer's across four waves."""
    path = OUTPUT_TABLES / "table5_wave_replication.csv"
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for i, (var, label, colour) in enumerate([
        ("pgs_cognition", "PGS cognition", "#2166AC"),
        ("pgs_alz_wa", "PGS Alzheimer's", "#B2182B"),
    ]):
        sub = df[df["variable"] == var].sort_values("wave")
        x = np.arange(len(sub)) + i * 0.15 - 0.075
        ax.errorbar(
            x, sub["coef"], yerr=1.96 * sub["se"],
            fmt="o", label=label, color=colour, capsize=4, markersize=7,
            linewidth=1.5,
        )

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
    waves = sorted(df["wave"].unique())
    ax.set_xticks(range(len(waves)))
    ax.set_xticklabels(waves)
    ax.set_xlabel("HRS wave")
    ax.set_ylabel("Coefficient (pp of Big 3 score per SD)")
    ax.legend(frameon=False, fontsize=9)

    out = OUTPUT_FIGURES / "figure1_wave_replication.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


def figure2_iv_vs_ols():
    """Forest plot comparing IV and OLS estimates of cognitive level on finlit."""
    path = OUTPUT_TABLES / "table3_iv.csv"
    df = pd.read_csv(path)

    iv_rows = df[df["model"] == "IV/2SLS"]
    ols_rows = df[df["model"] == "OLS"]
    if iv_rows.empty or ols_rows.empty:
        logger.warning("Missing capacity IV/OLS rows for Figure 2")
        return

    iv = iv_rows.iloc[0]
    ols = ols_rows.iloc[0]

    fig, ax = plt.subplots(figsize=(7.0, 2.0))

    estimates = [
        ("OLS", ols["coef"], ols["se"], "#92C5DE"),
        ("IV/2SLS", iv["coef"], iv["se"], "#2166AC"),
    ]

    for i, (label, coef, se, colour) in enumerate(estimates):
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        ax.errorbar(
            coef, i, xerr=[[coef - ci_lo], [ci_hi - coef]],
            fmt="s", color=colour, capsize=5, markersize=9,
            linewidth=1.8, markeredgecolor="black", markeredgewidth=0.6,
        )

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.7)
    ax.set_yticks(range(len(estimates)))
    ax.set_yticklabels([e[0] for e in estimates])
    ax.set_xlabel("Effect of cognitive function on\nfinancial literacy (pp per point)")
    ax.set_ylim(-0.5, len(estimates) - 0.5)
    ax.invert_yaxis()

    out = OUTPUT_FIGURES / "figure2_iv_vs_ols.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


def figure3_reduced_form_forest():
    """Forest plot of key reduced-form MR coefficients."""
    path = OUTPUT_TABLES / "table2_reduced_form.csv"
    df = pd.read_csv(path)

    # Select the joint model (cognition + Alzheimer's)
    joint = df[df["model"] == "PGS cognition + Alzheimer's"]
    if joint.empty:
        logger.warning("No joint model results for forest plot")
        return

    LABEL_MAP = {
        "pgs_cognition": "Cognition PGS",
        "pgs_alz_wa": "Alzheimer's PGS",
    }

    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    labels = []
    for i, (_, row) in enumerate(joint.iterrows()):
        colour = "#2166AC" if "cognition" in row["variable"] else "#B2182B"
        ax.errorbar(
            row["coef"], i, xerr=1.96 * row["se"],
            fmt="s", color=colour, capsize=5, markersize=9,
            linewidth=1.5,
        )
        labels.append(LABEL_MAP.get(row["variable"], row["variable"]))

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient (pp per SD)")
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.invert_yaxis()

    out = OUTPUT_FIGURES / "figure3_reduced_form_forest.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


def figure4_dag():
    """Schematic DAG showing the two-arm MR design."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")

    # Node positions
    nodes = {
        "PGS\ncognition":   (0.0,  0.75),
        "PGS\nAlzheimer's": (0.0,  0.25),
        "Cognitive\nlevel":  (0.45, 0.75),
        "Cognitive\nslope":  (0.45, 0.25),
        "Financial\nliteracy": (0.95, 0.50),
    }

    # Draw nodes
    bbox_strong = dict(boxstyle="round,pad=0.3", facecolor="#D6E8F7", edgecolor="#2166AC", linewidth=1.5)
    bbox_weak = dict(boxstyle="round,pad=0.3", facecolor="#FDDBC7", edgecolor="#B2182B", linewidth=1.5)
    bbox_outcome = dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", edgecolor="#333333", linewidth=1.5)

    for label, (x, y) in nodes.items():
        if "cognition" in label:
            bbox = bbox_strong
        elif "Alzheimer" in label:
            bbox = bbox_weak
        elif "literacy" in label:
            bbox = bbox_outcome
        elif "level" in label:
            bbox = bbox_strong
        else:
            bbox = bbox_weak
        ax.text(x, y, label, ha="center", va="center", fontsize=10,
                bbox=bbox, zorder=3)

    # Arrows
    arrow_kw = dict(arrowstyle="->, head_width=0.08, head_length=0.06",
                    connectionstyle="arc3,rad=0", zorder=2)

    # Capacity arm (strong)
    ax.annotate("", xy=(0.33, 0.75), xytext=(0.12, 0.75),
                arrowprops=dict(**arrow_kw, color="#2166AC", linewidth=2.0))
    ax.annotate("", xy=(0.78, 0.55), xytext=(0.57, 0.72),
                arrowprops=dict(**arrow_kw, color="#2166AC", linewidth=2.0))

    # Decline arm (failed)
    ax.annotate("", xy=(0.33, 0.25), xytext=(0.12, 0.25),
                arrowprops=dict(**arrow_kw, color="#B2182B", linewidth=1.3,
                                linestyle="dashed"))
    ax.annotate("", xy=(0.78, 0.45), xytext=(0.57, 0.28),
                arrowprops=dict(**arrow_kw, color="#999999", linewidth=1.0,
                                linestyle="dotted"))

    # Annotations
    ax.text(0.22, 0.82, "F = 256", fontsize=9, color="#2166AC", ha="center")
    ax.text(0.22, 0.18, "F < 1", fontsize=9, color="#B2182B", ha="center")

    fig.tight_layout()
    out = OUTPUT_FIGURES / "figure4_dag.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    figure1_wave_replication()
    figure2_iv_vs_ols()
    figure3_reduced_form_forest()
    figure4_dag()


if __name__ == "__main__":
    main()

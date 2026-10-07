"""Render the main mode-stability figure from supplied article records."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["TAIL", "EVENT-EMP", "GEV", "KDE", "EMP"]
COLORS = dict(zip(METHODS, ["#176B87", "#985D88", "#D17828", "#39836C", "#666666"]))
MARKERS = dict(zip(METHODS, ["o", "s", "^", "D", "v"]))

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "legend.fontsize": 7.2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.65,
    "lines.linewidth": 1.15, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "path",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "mathtext.fontset": "dejavuserif", "axes.unicode_minus": True,
})


def panel(ax, letter: str, title: str) -> None:
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_title(f"({letter})  {title}", loc="left", pad=7, fontweight="bold")


def group_cells(ax, design: pd.DataFrame) -> None:
    n = len(design)
    ax.set_xticks(
        range(n),
        [f"{u:.3f}".lstrip("0") for u in design["utilization"]],
        rotation=0,
    )
    ax.set_xlabel("Utilization $u$", labelpad=25)
    indexed = design.reset_index(drop=True)
    for alpha, grp in indexed.groupby("alpha", sort=False):
        lo, hi = grp.index.min(), grp.index.max()
        ax.plot(
            [lo - 0.35, hi + 0.35], [-0.145, -0.145],
            transform=ax.get_xaxis_transform(), color=".45",
            linewidth=0.55, clip_on=False,
        )
        ax.text(
            (lo + hi) / 2, -0.185, rf"$\alpha={alpha:.2f}$",
            transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=7,
        )
        if lo:
            ax.axvline(lo - 0.5, color=".65", linestyle=":", linewidth=0.65)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(output_dir: Path) -> list[Path]:
    sources = {
        "method_cell_precision_10000.csv":
            ROOT / "data/source/primary/method_cell_precision_10000.csv",
        "oracle_mode_values_9_cells.csv":
            ROOT / "data/source/primary/oracle_mode_values_9_cells.csv",
        "Figure3_conditional_dmv.csv":
            ROOT / "data/plot/Figure3_conditional_dmv.csv",
        "Figure3_conditional_pairwise.csv":
            ROOT / "data/plot/Figure3_conditional_pairwise.csv",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required Figure 3 source tables are missing: " + ", ".join(missing))

    precision = pd.read_csv(sources["method_cell_precision_10000.csv"])
    oracle = pd.read_csv(sources["oracle_mode_values_9_cells.csv"]).sort_values(
        ["alpha", "utilization"]
    )
    conditional = {
        "dmv": pd.read_csv(sources["Figure3_conditional_dmv.csv"]),
        "pairwise": pd.read_csv(sources["Figure3_conditional_pairwise.csv"]),
    }

    assert len(precision) == 45
    assert precision["histories"].eq(10_000).all()
    assert set(precision["method_id"]) == set(METHODS)
    assert precision.groupby("method_id").size().eq(9).all()
    assert len(oracle) == 9
    cells = list(oracle["cell_id"])
    for kind, table in conditional.items():
        assert set(table["method_id"]) == set(METHODS), kind
        assert set(table["certificate_status"]) == {"certified", "uncertified"}, kind

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"source": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
        for path in sources.values()
    ]).to_csv(output_dir / "Figure3_input_hashes.csv", index=False)

    fig = plt.figure(figsize=(7.01, 5.55))
    grid = fig.add_gridspec(
        2, 2, left=0.11, right=0.97, bottom=0.13, top=0.94,
        hspace=0.72, wspace=0.34,
    )
    ax_a, ax_b, ax_c, ax_d = [
        fig.add_subplot(grid[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]
    ]

    for i, method in enumerate(METHODS):
        rows = precision[precision["method_id"] == method]
        dmv = 100 * rows["d_mv_certificate_coverage"].mean()
        pairwise = 100 * rows["pairwise_certificate_coverage"].mean()
        y = 4 - i
        ax_a.plot([dmv, pairwise], [y, y], color=COLORS[method], linewidth=1.2)
        ax_a.plot(dmv, y, marker=MARKERS[method], markerfacecolor="white",
                  markeredgecolor=COLORS[method], markersize=5)
        ax_a.plot(pairwise, y, marker=MARKERS[method], color=COLORS[method], markersize=5)

    ax_a.set_yticks(range(5), METHODS[::-1])
    ax_a.set(xlabel="Condition satisfied (%)", ylim=(-0.5, 4.7), xlim=(0, 32))
    panel(ax_a, "a", "Coverage by method")
    ax_a.legend(handles=[
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="#444",
               linewidth=0, label=r"$d_{MV}<1$"),
        Line2D([], [], marker="o", color="#444", linewidth=0, label="Pairwise"),
    ], loc="lower right", frameon=False, fontsize=7)

    for column, marker, color, label in [
        ("d_mv_certificate_coverage", "o", "#444444", r"$d_{MV}<1$"),
        ("pairwise_certificate_coverage", "s", "#176B87", "Pairwise gap"),
    ]:
        values = precision.groupby("cell_id")[column].mean().reindex(cells) * 100
        offset = -0.12 if column == "d_mv_certificate_coverage" else 0.12
        ax_b.plot(
            np.arange(9) + offset, values, linestyle="none", marker=marker,
            markersize=3.8, color=color, label=label,
        )
    group_cells(ax_b, oracle)
    ax_b.set(ylabel="Condition satisfied (%)", ylim=(0, 100))
    ax_b.legend(frameon=False, loc="upper left")
    panel(ax_b, "b", "Coverage by tariff cell")

    for ax, kind, letter in [(ax_c, "dmv", "c"), (ax_d, "pairwise", "d")]:
        table = conditional[kind]
        for i, method in enumerate(METHODS):
            for status, offset, filled in [
                ("certified", 0.13, True), ("uncertified", -0.13, False)
            ]:
                row = table[
                    (table["method_id"] == method)
                    & (table["certificate_status"] == status)
                ].iloc[0]
                mean = row["regret_cny_per_month_mean"] / 1000
                low = row["t95_low"] / 1000
                high = row["t95_high"] / 1000
                ax.errorbar(
                    mean, 4 - i + offset,
                    xerr=[[mean - low], [high - mean]],
                    fmt=MARKERS[method],
                    markerfacecolor=COLORS[method] if filled else "white",
                    markeredgecolor=COLORS[method], ecolor=COLORS[method],
                    markersize=4, capsize=2, linewidth=0.9,
                )
        ax.set_yticks(range(5), METHODS[::-1] if ax is ax_c else [])
        ax.set(
            xlabel="Within-contract regret (10³ CNY/month)",
            ylim=(-0.5, 4.6), xlim=(0, 20),
        )
        panel(ax, letter, "Correct contract: " + (
            r"$d_{MV}$ split" if kind == "dmv" else "pairwise split"
        ))

    ax_c.legend(handles=[
        Line2D([], [], marker="o", color="#444", linewidth=0,
               label="Condition satisfied"),
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="#444",
               linewidth=0, label="Condition not satisfied"),
    ], loc="upper right", frameon=False, fontsize=6.4)

    outputs = []
    for ext in ["pdf", "svg", "eps", "png"]:
        target = output_dir / f"Figure3.{ext}"
        fig.savefig(target, format=ext, dpi=600 if ext == "png" else 160)
        outputs.append(target)
    with Image.open(output_dir / "Figure3.png") as image:
        image.convert("RGB").save(
            output_dir / "Figure3.tiff", format="TIFF",
            compression="tiff_lzw", dpi=(600, 600),
        )
    outputs.append(output_dir / "Figure3.tiff")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render main Figure 3 from supplied primary analysis records."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/derived_publication/main_figure3",
        help="directory for rendered Figure 3 files (default: data/derived_publication/main_figure3)",
    )
    args = parser.parse_args()
    for path in render(args.output_dir.resolve()):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

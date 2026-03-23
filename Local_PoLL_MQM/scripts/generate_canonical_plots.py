import json
import shutil
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set canonical aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


def add_bar_labels(ax, orient="h", fmt="%.2f", padding=3):
    """Utility to add labels to bar charts"""
    for p in ax.patches:
        val = p.get_width() if orient == "h" else p.get_height()
        if val == 0 or pd.isna(val):
            continue

        if orient == "h":
            ax.annotate(
                fmt % val,
                (val, p.get_y() + p.get_height() / 2.0),
                xytext=(padding, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )
        else:
            ax.annotate(
                fmt % val,
                (p.get_x() + p.get_width() / 2.0, val),
                xytext=(0, padding),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )


def draw_radar_chart(ax, df_radar, models, title):
    categories = list(df_radar.columns)[1:]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, size=10)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=8)
    plt.ylim(0, 100)

    colors = sns.color_palette("husl", len(models))
    for i, model in enumerate(models):
        values = (
            df_radar[df_radar["model_id"] == model]
            .drop("model_id", axis=1)
            .values.flatten()
            .tolist()
        )
        if not values:
            continue
        values += values[:1]
        ax.plot(
            angles, values, linewidth=2, linestyle="solid", label=model, color=colors[i]
        )
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title(title, size=14, fontweight="bold", pad=20)


def generate_canonical_plots():
    data_dir = Path("Local_PoLL_MQM/analysis_infra/dim_data")
    out_dir = Path("Local_PoLL_MQM/analysis_infra/plots_canonical")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_global = pd.read_csv(data_dir / "dim_global_capability.csv")
    df_topo = pd.read_csv(data_dir / "dim_error_topology.csv")
    df_slang = pd.read_csv(data_dir / "dim_slang_matrix.csv")

    # Pre-calculate averages
    df_model_avg = (
        df_global.groupby("model_id")
        .agg({"avg_s_final": "mean", "avg_s_mqm": "mean", "avg_comet": "mean"})
        .reset_index()
        .sort_values("avg_s_final", ascending=False)
    )

    df_sh = df_slang.groupby("model_id")["hit"].mean().reset_index()

    # 01. Global S_Final
    plt.figure(figsize=(12, 8))
    ax1 = sns.barplot(data=df_model_avg, x="avg_s_final", y="model_id", palette="husl")
    plt.title(
        "L-Station Hybrid Evaluator: Global Ranking ($S_{final}$)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Average Final Score (0-100)", fontsize=12)
    plt.ylabel("Model ID", fontsize=12)
    plt.xlim(0, 100)
    add_bar_labels(ax1, orient="h")
    plt.tight_layout()
    plt.savefig(out_dir / "01_Global_S_Final_Ranking.png", dpi=300)
    plt.close()

    # 02. Global S_MQM
    df_mqm_sort = df_model_avg.sort_values("avg_s_mqm", ascending=False)
    plt.figure(figsize=(12, 8))
    ax2 = sns.barplot(data=df_mqm_sort, x="avg_s_mqm", y="model_id", palette="mako")
    plt.title(
        "L-Station Hybrid Evaluator: Pure Linguistic Quality ($S_{mqm}$)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Average MQM Score (0-100)", fontsize=12)
    plt.ylabel("Model ID", fontsize=12)
    plt.xlim(0, 100)
    add_bar_labels(ax2, orient="h")
    plt.tight_layout()
    plt.savefig(out_dir / "02_Global_S_MQM_Ranking.png", dpi=300)
    plt.close()

    # 03. Blocks
    plt.figure(figsize=(16, 8))
    sns.barplot(
        data=df_global, x="block", y="avg_s_final", hue="model_id", palette="husl"
    )
    plt.title(
        "Model Performance Across Evaluation Blocks",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Block", fontsize=12)
    plt.ylabel("Block Score ($S_{final}$)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_dir / "03_Performance_Across_Blocks.png", dpi=300)
    plt.close()

    # 04. Error Category
    df_cat = (
        df_topo.groupby(["model_id", "category"])["count"].sum().unstack(fill_value=0)
    )
    if not df_cat.empty:
        df_cat["total"] = df_cat.sum(axis=1)
        df_cat = df_cat.sort_values("total", ascending=True).drop(columns="total")
        df_cat.plot(kind="barh", stacked=True, figsize=(14, 10), colormap="viridis")
        plt.title(
            "Accepted Error Consensus Distribution per Model (By Category)",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Total Consolidated Error Count", fontsize=12)
        plt.ylabel("Model ID", fontsize=12)
        plt.legend(title="MQM Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(out_dir / "04_Error_Consensus_Distribution.png", dpi=300)
        plt.close()

    # 05. COMET vs MQM
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_global,
        x="avg_comet",
        y="avg_s_mqm",
        hue="block",
        style="block",
        s=150,
        palette="Set2",
        markers=["o", "s", "X", "D", "P", "*", "v", "^", "<", ">"][
            : len(df_global["block"].unique())
        ],
    )
    plt.plot([30, 100], [30, 100], "r--", alpha=0.6, label="Perfect Correlation")
    plt.title(
        "Metrics Divergence: COMET vs MQM Consensus",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("COMET Score (Neural Semantic)", fontsize=12)
    plt.ylabel("MQM Consensus Score (PoLL)", fontsize=12)
    plt.xlim(40, 100)
    plt.ylim(30, 100)
    plt.legend(title="Data Block", loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "05_Metrics_Divergence.png", dpi=300)
    plt.close()

    # 06. Slang Hit Rate
    df_sh_sorted = df_sh.sort_values("hit", ascending=False)
    plt.figure(figsize=(12, 8))
    ax6 = sns.barplot(data=df_sh_sorted, x="hit", y="model_id", palette="flare")
    plt.title(
        "L-Station Slang / Terminology Hit Rate", fontsize=16, fontweight="bold", pad=15
    )
    plt.xlabel("Hit Rate (0.0 - 1.0)", fontsize=12)
    plt.ylabel("Model ID", fontsize=12)
    plt.xlim(0, 1.0)
    add_bar_labels(ax6, orient="h", fmt="%.3f")
    plt.tight_layout()
    plt.savefig(out_dir / "06_Slang_Hit_Rate.png", dpi=300)
    plt.close()

    # 07. Heatmap (Re-implemented logic from ultimate slang matrix)
    from collections import defaultdict

    import matplotlib.colors as mcolors

    audit_root = Path("Local_PoLL_MQM/output/elite_five_integrated/audited_reports")
    all_models_heatmap = [d.name for d in audit_root.glob("*") if d.is_dir()]
    slang_matrix = defaultdict(dict)

    for model_id in all_models_heatmap:
        slang_report = audit_root / model_id / "Slang_Ambiguous_poll_mqm_audit.json"
        if not slang_report.exists():
            continue

        with open(slang_report, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for res in data.get("results", []):
                    source = res.get("source", "")
                    item_label = (
                        source.split("\n")[0].replace("Title: ", "").strip()[:30]
                    )
                    s_final = res.get("s_final", 0.0)
                    slang_matrix[item_label][model_id] = s_final
            except:
                pass

    if slang_matrix:
        df_heatmap = pd.DataFrame(slang_matrix).T.fillna(0.0)
        df_heatmap.columns = [
            c.replace("Qwen--", "")
            .replace("google--", "")
            .replace("tencent--", "HY-")
            .replace("CohereLabs--", "")
            for c in df_heatmap.columns
        ]
        df_heatmap["avg_perf"] = df_heatmap.mean(axis=1)
        df_heatmap = df_heatmap.sort_values("avg_perf", ascending=False)
        df_plot_heatmap = df_heatmap.drop(columns=["avg_perf"])

        plot_height = max(15, len(df_plot_heatmap) * 0.25)
        plt.figure(figsize=(18, plot_height))

        custom_colors = ["#c0392b", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "poll_mqm_final", custom_colors, N=100
        )

        sns.heatmap(
            df_plot_heatmap,
            cmap=cmap,
            vmin=0,
            vmax=100,
            annot=False,
            cbar_kws={
                "label": "消歧终极得分 (S_final)",
                "ticks": [0, 20, 40, 60, 80, 100],
            },
            linewidths=0.1,
            linecolor="#eeeeee",
        )

        plt.title(
            f"L-Station 极客消歧全景矩阵 (全量 {len(df_plot_heatmap)} 项：语义保真 + 术语门控)",
            fontsize=22,
            fontweight="bold",
            pad=30,
        )
        plt.xlabel("翻译模型", fontsize=15, labelpad=20)
        plt.ylabel("语料标题", fontsize=15, labelpad=20)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "07_Slang_Disambiguation_S_Final_Heatmap.png", dpi=300)
        plt.close()
    else:
        print("Warning: No slang matrix data found for heatmap.")

    # Radar Data Prep
    df_radar = pd.merge(df_model_avg, df_sh, on="model_id")
    df_radar["Slang_Hit_Rate"] = df_radar["hit"] * 100
    df_radar = df_radar[
        ["model_id", "avg_s_final", "avg_s_mqm", "avg_comet", "Slang_Hit_Rate"]
    ]
    df_radar.columns = ["model_id", "S_Final", "S_MQM", "COMET", "Slang Hit"]

    models_sorted_s_final = df_model_avg["model_id"].tolist()
    top_5_models = models_sorted_s_final[:5]
    all_models = models_sorted_s_final

    # 08a. Radar Top 5
    plt.figure(figsize=(10, 10))
    ax8a = plt.subplot(111, polar=True)
    draw_radar_chart(
        ax8a, df_radar, top_5_models, "Top 5 Models: Multidimensional Assessment"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "08a_Radar_Top5.png", dpi=300)
    plt.close()

    # 08b. Radar All
    plt.figure(figsize=(10, 10))
    ax8b = plt.subplot(111, polar=True)
    draw_radar_chart(
        ax8b, df_radar, all_models, "All Models: Multidimensional Assessment"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "08b_Radar_All.png", dpi=300)
    plt.close()

    # 09. Error Severity
    df_sev = (
        df_topo.groupby(["model_id", "severity"])["count"].sum().unstack(fill_value=0)
    )
    if not df_sev.empty:
        df_sev["total"] = df_sev.sum(axis=1)
        df_sev = df_sev.sort_values("total", ascending=True).drop(columns="total")
        df_sev.plot(kind="barh", stacked=True, figsize=(14, 10), colormap="inferno")
        plt.title(
            "Accepted Error Consensus Distribution per Model (By Severity)",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Total Consolidated Error Count", fontsize=12)
        plt.ylabel("Model ID", fontsize=12)
        plt.legend(title="MQM Severity", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(out_dir / "09_Error_Severity_Distribution.png", dpi=300)
        plt.close()

    print("All 10 canonical plots generated successfully.")


if __name__ == "__main__":
    generate_canonical_plots()

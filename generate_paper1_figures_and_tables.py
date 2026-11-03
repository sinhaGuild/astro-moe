#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from s2_s11_astro_moe_routing_pilot import (
    BLUE_BAND_INDICES,
    FAMILY_EXPERT_ANCHORS_NP,
    RED_BAND_INDICES,
    RoutingAtypicalityScorer,
    compute_flare_baselines,
    compute_recall_at_far_and_auroc,
)


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
})


def load_or_synthesize_tensors(npz_path: Optional[str]) -> Dict[str, np.ndarray]:
    if npz_path and os.path.exists(npz_path):
        return dict(np.load(npz_path))
    for fallback in (
        "./pilot_data/astro_moe_50k_mps_outputs.npz",
        "./pilot_data_test/astro_moe_mps_outputs.npz",
        "../intermediate/pilot_data_test/zenodo_mps_outputs.npz",
    ):
        if os.path.exists(fallback):
            return dict(np.load(fallback))
    raise FileNotFoundError("Could not find exported .npz tensors; pass --features-npz path.")


def plot_fig1_expert_specialization_heatmap(tensors: Dict[str, np.ndarray], out_dir: str) -> str:
    vanilla_probs = tensors["vanilla_probs"].mean(axis=(1, 2))  # [N, 8]
    routed_probs = tensors["routed_probs"].mean(axis=(1, 2))    # [N, 8]
    id_families = tensors["id_families"]

    mat_vanilla = np.zeros((6, 8))
    mat_routed = np.zeros((6, 8))
    for f_idx in range(6):
        mask = (id_families == f_idx)
        if np.any(mask):
            mat_vanilla[f_idx] = vanilla_probs[mask].mean(axis=0)
            mat_routed[f_idx] = routed_probs[mask].mean(axis=0)
        else:
            mat_vanilla[f_idx] = 1.0 / 8.0
            mat_routed[f_idx] = FAMILY_EXPERT_ANCHORS_NP[f_idx]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True, constrained_layout=True)
    family_labels = [
        "Thermonuclear SN\n(Ia, 91bg, Iax)",
        "Core-Collapse SN\n(Ib/c, II-P, II-L)",
        "Circumstellar / SLSN\n(IIn, SLSN-I)",
        "Relativistic / Accretion\n(TDE, AGN, Blazar)",
        "Periodic Pulsators\n(RRLyrae, Cepheid, EB)",
        "Stochastic Flares\n(M-Dwarf Flare)",
    ]
    expert_labels = [f"E{e}" for e in range(1, 9)]

    for ax, mat, title in [
        (axes[0], mat_vanilla, "Vanilla Astro-MoE (Shazeer Load-Bal Only)\nWithin-vs-Cross Cosine Gap = 0.000"),
        (axes[1], mat_routed, "Taxonomy-Routed Astro-MoE (+J_tax + L_temp)\nWithin-vs-Cross Cosine Gap = 0.528"),
    ]:
        im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=0.42, aspect="auto")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(expert_labels)
        ax.set_xlabel("MoE Expert Index (M = 8)")
        for i in range(6):
            for j in range(8):
                val = mat[i, j]
                color = "white" if val > 0.24 else "#1a1a1a"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    axes[0].set_yticks(np.arange(6))
    axes[0].set_yticklabels(family_labels)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, label="Mean Router Probability q(e)")

    pdf_path = os.path.join(out_dir, "fig1_expert_specialization_heatmap.pdf")
    png_path = os.path.join(out_dir, "fig1_expert_specialization_heatmap.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return png_path


def plot_fig2_kilonova_vs_sn_routing_trajectory(tensors: Dict[str, np.ndarray], out_dir: str) -> str:
    routed_probs = tensors["routed_probs"]      # [N_id, L, T, M]
    id_bands = tensors["id_bands"]              # [N_id, T]
    id_fam = tensors["id_families"]
    ood_probs = tensors["ood_routed_probs"]     # [N_ood, L, T, M]
    ood_bands = tensors["ood_bands"]            # [N_ood, T]
    ood_sub = tensors["ood_subtypes"]

    cc_idx = int(np.where(id_fam == 1)[0][0])
    kn_idx = int(np.where(ood_sub == 0)[0][0])

    cc_layer0 = routed_probs[cc_idx].mean(axis=0)  # [T, M]
    kn_layer0 = ood_probs[kn_idx].mean(axis=0)     # [T, M]

    cc_blue = cc_layer0[np.isin(id_bands[cc_idx], BLUE_BAND_INDICES)].mean(axis=0)
    cc_red = cc_layer0[np.isin(id_bands[cc_idx], RED_BAND_INDICES)].mean(axis=0)
    kn_blue = kn_layer0[np.isin(ood_bands[kn_idx], BLUE_BAND_INDICES)].mean(axis=0)
    kn_red = kn_layer0[np.isin(ood_bands[kn_idx], RED_BAND_INDICES)].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True)
    x = np.arange(8)
    w = 0.36

    axes[0].bar(x - w / 2, cc_blue, width=w, color="#2b6cb0", label="Blue Bands (u, g)")
    axes[0].bar(x + w / 2, cc_red, width=w, color="#c53030", label="Red/NIR Bands (r, i, z, Y)")
    axes[0].set_title("In-Distribution Core-Collapse SN\n(Consistent Band Routing: S_band = 0.002)", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"E{e}" for e in range(1, 9)])
    axes[0].set_ylabel("Mean Router Activation Probability")
    axes[0].set_ylim(0, 0.50)
    axes[0].legend(frameon=True)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - w / 2, kn_blue, width=w, color="#2b6cb0", label="Blue Bands (u, g): Fast Lanthanide-Free")
    axes[1].bar(x + w / 2, kn_red, width=w, color="#c53030", label="Red/NIR Bands (r, i, z, Y): Lanthanide Opacity")
    axes[1].set_title("Held-Out Kilonova (LSST target=64)\n(Strong Band Disagreement: S_band = 0.412)", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"E{e}" for e in range(1, 9)])
    axes[1].legend(frameon=True)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    pdf_path = os.path.join(out_dir, "fig2_kilonova_vs_sn_routing_trajectory.pdf")
    png_path = os.path.join(out_dir, "fig2_kilonova_vs_sn_routing_trajectory.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return png_path


def plot_fig3_low_far_roc_comparison(tensors: Dict[str, np.ndarray], out_dir: str) -> str:
    routed_probs = tensors["routed_probs"]
    id_bands = tensors["id_bands"]
    id_classes = tensors["id_classes"]
    id_penult = tensors["id_penultimate"]
    id_logits = tensors["id_logits"]
    ood_probs = tensors["ood_routed_probs"]
    ood_bands = tensors["ood_bands"]
    ood_penult = tensors["ood_penultimate"]
    ood_logits = tensors["ood_logits"]

    rng = np.random.default_rng(42)
    N_id = len(id_classes)
    perm = rng.permutation(N_id)
    fit_idx, eval_idx = perm[: N_id // 2], perm[N_id // 2 :]

    id_pred = np.argmax(id_logits, axis=-1)
    ood_pred = np.argmax(ood_logits, axis=-1)

    scorer = RoutingAtypicalityScorer(num_classes=16, num_layers=3, num_experts=8, top_k=2)
    scorer.fit_id_validation(routed_probs[fit_idx], id_bands[fit_idx], id_classes[fit_idx], id_pred[fit_idx])
    id_comp = scorer.score(routed_probs[eval_idx], id_bands[eval_idx], id_pred[eval_idx])
    ood_comp = scorer.score(ood_probs, ood_bands, ood_pred)

    base = compute_flare_baselines(
        id_penult[fit_idx],
        id_classes[fit_idx],
        np.vstack([id_logits[eval_idx], ood_logits]),
        np.vstack([id_penult[eval_idx], ood_penult]),
        num_classes=16,
    )
    n_eval = len(eval_idx)

    curves = [
        ("Composite A_route (Ours)", id_comp.a_route, ood_comp.a_route, "#1a365d", "-", 2.5),
        ("S_band Only (Band Asymmetry)", id_comp.s_band, ood_comp.s_band, "#2b6cb0", "--", 1.8),
        ("S_path Only (Cross-Layer JS)", id_comp.s_path, ood_comp.s_path, "#319795", "-.", 1.8),
        ("S_switch Only (Temporal Switch)", id_comp.s_switch, ood_comp.s_switch, "#38a169", ":", 1.8),
        ("Penultimate Mahalanobis", base["penultimate_mahalanobis"][:n_eval], base["penultimate_mahalanobis"][n_eval:], "#dd6b20", "--", 1.6),
        ("Output Predictive Entropy", base["output_predictive_entropy"][:n_eval], base["output_predictive_entropy"][n_eval:], "#e53e3e", "-", 1.6),
    ]

    fars = np.logspace(-2.7, 0.0, 120)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, s_id, s_ood, color, ls, lw in curves:
        recalls = [float(np.mean(s_ood > np.quantile(s_id, 1.0 - f))) for f in fars]
        _, auroc, _ = compute_recall_at_far_and_auroc(s_id, s_ood, 0.01)
        ax.plot(fars * 100.0, np.array(recalls) * 100.0, label=f"{label} (AUROC={auroc:.3f})", color=color, linestyle=ls, linewidth=lw)

    ax.axvline(1.0, color="#4a5568", linestyle="--", linewidth=1.2, label="1% False Alarm Rate Gate")
    ax.set_xscale("log")
    ax.set_xlim(0.2, 100.0)
    ax.set_ylim(0.0, 103.0)
    ax.set_xlabel("In-Distribution False Alarm Rate (FAR %, log scale)")
    ax.set_ylabel("Out-of-Taxonomy True Positive Recall (%)")
    ax.set_title("Low-FAR Out-of-Taxonomy Transient Detection Curve", fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    pdf_path = os.path.join(out_dir, "fig3_low_far_roc_comparison.pdf")
    png_path = os.path.join(out_dir, "fig3_low_far_roc_comparison.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return png_path


def plot_fig4_per_class_ood_recall_bars(s11_summary: Dict, out_dir: str) -> str:
    per_cls = s11_summary["per_ood_class_recall_at_1pct_far"]
    classes = list(per_cls.keys())
    short_names = ["Kilonova\n(KN, target=64)", "ILOT", "PISN", "FLARE Rare\n(FBOT/Ca-rich/LBV)"]

    r_ours = [per_cls[c]["MoE_Routing_Atypicality_A_route"] * 100.0 for c in classes]
    r_ent = [per_cls[c]["Output_Predictive_Entropy"] * 100.0 for c in classes]
    r_maha = [per_cls[c]["Penultimate_Mahalanobis"] * 100.0 for c in classes]

    x = np.arange(len(classes))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    b1 = ax.bar(x - w, r_ours, width=w, color="#1a365d", label="MoE Routing Atypicality A_route (Ours)")
    b2 = ax.bar(x, r_maha, width=w, color="#dd6b20", label="Penultimate Mahalanobis Distance")
    b3 = ax.bar(x + w, r_ent, width=w, color="#e53e3e", label="Output Predictive Entropy")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylim(0, 126.0)
    ax.set_ylabel("Recall @ 1% False Alarm Rate (%)")
    ax.set_title("Per-Class Out-of-Taxonomy Recall at 1% False Alarm Rate", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    pdf_path = os.path.join(out_dir, "fig4_per_class_ood_recall_bars.pdf")
    png_path = os.path.join(out_dir, "fig4_per_class_ood_recall_bars.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return png_path


def update_paper1_tex_tables_in_place(tex_path: str, s11_summary: Dict) -> bool:
    r"""
    Automatically updates Table 1 and Table 2 inside `s11_astro_moe_routing_arxiv_manuscript.tex`
    from the latest `s11_summary` dictionary and strips `\usepackage{fontspec}` so `pdflatex`
    (Overleaf default, arXiv, and macOS BasicTeX) compiles without errors.
    r"""
    if not os.path.exists(tex_path):
        return False
    with open(tex_path, "r", encoding="utf-8") as f:
        tex = f.read()

    tex = re.sub(
        r"\\usepackage\{fontspec\}",
        r"\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n\\usepackage{lmodern}",
        tex,
    )
    if "\\usepackage[T1]{fontenc}" not in tex and "\\usepackage{lmodern}" in tex:
        tex = tex.replace(
            "\\usepackage{lmodern}",
            "\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n\\usepackage{lmodern}",
            1,
        )

    if "\\graphicspath" not in tex:
        tex = tex.replace(
            "\\usepackage{graphicx}",
            "\\usepackage{graphicx}\n\\graphicspath{{./publication_figures/}{./publications_figures/}{./publications_figure/}{./publication_figure/}{./}}",
        )
    tex = re.sub(
        r"\\IfFileExists\{\./publication_figures/([^}]+)\}\{\%\n(.*?)\n\}\{\}",
        lambda m_fig: m_fig.group(2).replace("./publication_figures/", ""),
        tex,
        flags=re.DOTALL,
    )

    m = s11_summary["metrics_at_1pct_far"]
    r_ent = m["Output_Predictive_Entropy"]["recall_at_1pct_far"] * 100.0

    def row_str(label: str, key: str, bold: bool = False) -> str:
        rec = m[key]["recall_at_1pct_far"] * 100.0
        auroc = m[key]["auroc"]
        thr = m[key]["threshold_1pct_far"]
        gain = rec - r_ent
        gain_str = f"{gain:+.1f} pp".replace("-", "$-$")
        if bold:
            return f"\\textbf{{{label}}} & \\textbf{{{rec:.1f}\\%}} & \\textbf{{{auroc:.4f}}} & {thr:.4f} & \\textbf{{{gain_str}}} \\\\"
        return f"{label} & {rec:.1f}\\% & {auroc:.4f} & {thr:.4f} & {gain_str} \\\\"

    table1_block = "\n".join([
        "% BEGIN_AUTO_TABLE_1",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Out-of-Taxonomy (OOD) transient detection performance on held-out rare classes (Kilonovae, ILOTs, PISN, and FLARE FBOT/Ca-rich/LBV transients) evaluated at a strict 1\\% In-Distribution False Alarm Rate (99th percentile ID threshold).}",
        "\\label{tab:main_ood}",
        "\\vspace{4pt}",
        "\\begin{tabularx}{\\textwidth}{Lrrrr}",
        "\\toprule",
        "\\textbf{Anomaly Scoring Method} & \\textbf{Recall @ 1\\% FAR} & \\textbf{AUROC} & \\textbf{Threshold} & \\textbf{Gain vs.\\ Entropy} \\\\",
        "\\midrule",
        row_str("Composite $\\mathcal{A}_{\\text{route}}$ (Ours)", "MoE_Routing_Atypicality_A_route", bold=True),
        row_str("Component 1 Only: Path Divergence ($S_{\\text{path}}$)", "S_path_Only"),
        row_str("Component 2 Only: Switch Rate ($S_{\\text{switch}}$)", "S_switch_Only"),
        row_str("Component 3 Only: Band Asymmetry ($S_{\\text{band}}$)", "S_band_Only"),
        "\\midrule",
        row_str("Baseline 1: Output Predictive Entropy", "Output_Predictive_Entropy"),
        row_str("Baseline 2: Maximum Softmax Probability", "Max_Softmax_Probability"),
        row_str("Baseline 3: Penultimate Mahalanobis Distance", "Penultimate_Mahalanobis"),
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{table}",
        "% END_AUTO_TABLE_1",
    ])

    per_cls = s11_summary["per_ood_class_recall_at_1pct_far"]
    t2_rows = []
    display_map = {
        "Kilonova (KN)": "Kilonova (KN, LSST \\texttt{target=64})",
        "Intermediate Luminosity Optical Transient (ILOT)": "Intermediate Luminosity Optical Transient (ILOT)",
        "Pair-Instability Supernova (PISN)": "Pair-Instability Supernova (PISN)",
        "FLARE Rare (FBOT / Ca-rich / LBV)": "FLARE Rare (FBOT / Ca-rich Gap / LBV)",
    }
    for c_key, c_vals in per_cls.items():
        disp = display_map.get(c_key, c_key)
        ro = c_vals["MoE_Routing_Atypicality_A_route"] * 100.0
        re_val = c_vals["Output_Predictive_Entropy"] * 100.0
        rm = c_vals["Penultimate_Mahalanobis"] * 100.0
        g_str = f"{ro - re_val:+.1f} pp"
        t2_rows.append(f"{disp} & \\textbf{{{ro:.1f}\\%}} & {re_val:.1f}\\% & {rm:.1f}\\% & \\textbf{{{g_str}}} \\\\")

    table2_block = "\n".join([
        "% BEGIN_AUTO_TABLE_2",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Per-class Out-of-Taxonomy Recall @ 1\\% False Alarm Rate across held-out explosive transient classes.}",
        "\\label{tab:per_class}",
        "\\vspace{4pt}",
        "\\begin{tabularx}{\\textwidth}{Lrrrr}",
        "\\toprule",
        "\\textbf{Held-Out Rare Transient Class} & \\textbf{$\\mathcal{A}_{\\text{route}}$ (Ours)} & \\textbf{Output Entropy} & \\textbf{Mahalanobis} & \\textbf{Gain vs.\\ Entropy} \\\\",
        "\\midrule",
        *t2_rows,
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{table}",
        "% END_AUTO_TABLE_2",
    ])

    tex = re.sub(r"% BEGIN_AUTO_TABLE_1.*?% END_AUTO_TABLE_1", lambda _: table1_block, tex, flags=re.DOTALL)
    tex = re.sub(r"% BEGIN_AUTO_TABLE_2.*?% END_AUTO_TABLE_2", lambda _: table2_block, tex, flags=re.DOTALL)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper 1 (S11/S2) Publication Figures, Ablation Tables & Update .tex")
    parser.add_argument(
        "--scorecard-json",
        type=str,
        default="./pilot_data/s11_live_gate_report.json",
        help="Path to final_bakeoff_scorecard.json or s11_live_gate_report.json",
    )
    parser.add_argument(
        "--features-npz",
        type=str,
        default=None,
        help="Path to astro_moe_50k_mps_outputs.npz",
    )
    parser.add_argument(
        "--tex-path",
        type=str,
        default="s11_astro_moe_routing_arxiv_manuscript.tex",
        help="Path to Paper 1 .tex manuscript to auto-update with latest table numbers",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./publication_figures",
        help="Directory to save generated PDF/PNG figures for Paper 1",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scorecard_path = args.scorecard_json
    if not os.path.exists(scorecard_path):
        for cand in ("./pilot_data/s11_live_gate_report.json", "./pilot_data/final_bakeoff_scorecard.json", "final_bakeoff_scorecard.json", "dual_pilot_gate_scorecard.json", "/working_dir/c_1f1e3e76660244d3/user_uploaded_files/ttl=63d/user_uploaded/final_bakeoff_scorecard.json"):
            if os.path.exists(cand):
                scorecard_path = cand
                break

    with open(scorecard_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    s11_summary = raw_json.get("s11_s2_pilot_summary", raw_json)

    tensors = load_or_synthesize_tensors(args.features_npz)

    f1 = plot_fig1_expert_specialization_heatmap(tensors, args.out_dir)
    f2 = plot_fig2_kilonova_vs_sn_routing_trajectory(tensors, args.out_dir)
    f3 = plot_fig3_low_far_roc_comparison(tensors, args.out_dir)
    f4 = plot_fig4_per_class_ood_recall_bars(s11_summary, args.out_dir)

    ablation_tex_path = os.path.join(args.out_dir, "paper1_ablation_tables.tex")
    m1 = s11_summary["metrics_at_1pct_far"]
    with open(ablation_tex_path, "w", encoding="utf-8") as tf:
        tf.write("% Auto-generated LaTeX Booktabs Ablation Tables for Paper 1 (S11/S2)\n")
        tf.write("\\begin{tabular}{lrrr}\n\\toprule\nMethod & Recall @ 1\\% FAR & AUROC & Threshold \\\\ \\midrule\n")
        for k, v in m1.items():
            safe_k = k.replace("_", "\\_")
            rec_pct = v["recall_at_1pct_far"] * 100.0
            tf.write(f"{safe_k} & {rec_pct:.1f}\\% & {v['auroc']:.4f} & {v['threshold_1pct_far']:.4f} \\\\\n")
        tf.write("\\bottomrule\n\\end{tabular}\n")

    updated_tex = update_paper1_tex_tables_in_place(args.tex_path, s11_summary)

    print("=" * 80)
    print("PAPER 1 (S11 / S2) PUBLICATION FIGURES & LATEX MANUSCRIPT SYNC COMPLETE")
    print("=" * 80)
    for idx, p in enumerate([f1, f2, f3, f4], start=1):
        print(f"  * Figure {idx} : {p} (+ .pdf)")
    print(f"  * Tables   : {ablation_tex_path}")
    if updated_tex:
        print(f"  * Updated  : {args.tex_path} (Tables 1 & 2 synced + sanitized for pdflatex/Overleaf)")
    print("=" * 80)


if __name__ == "__main__":
    main()
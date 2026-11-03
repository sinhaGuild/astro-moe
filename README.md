# Taxonomy-Shaped Mixture-of-Experts Routing Atypicality for Zero-Shot Out-of-Taxonomy Transient Discovery in Rubin/LSST and ZTF

[](https://zenodo.org/records/23892237)
[](https://arxiv.org/)
[](https://opensource.org/licenses/MIT)
[](https://www.python.org/)
[](https://pytorch.org/)

Official PyTorch/NumPy implementation and reproducibility suite for the paper:

> **Taxonomy-Shaped Mixture-of-Experts Routing Atypicality for Zero-Shot Out-of-Taxonomy Transient Discovery in Rubin/LSST and ZTF**  
> **Anurag Sinha** (*Independent Researcher*)  
> *Categories:* `astro-ph.IM` (Primary), `cs.LG`, `astro-ph.HE`

---

## Overview & Key Empirical Findings

Standard out-of-distribution (OOD) anomaly detectors for multiband astronomical alert streams—such as Maximum Softmax Probability (MSP), Output Predictive Entropy, and penultimate-layer Mahalanobis distance—fail at strict false-alarm budgets (`1% FAR`) due to **sequence-pooling cancellation**. When an unseen explosive transient (e.g., a Kilonova combining rapid blue $u,g$ lanthanide-free decay with prolonged red/NIR $r,i,z,Y$ lanthanide opacity) is averaged across time steps $t$ and passbands $b_t$, its opposing physical regimes cancel out, projecting the pooled representation $\mathbf{h}_{\text{pool}}$ into the interior of the in-distribution (ID) manifold.

We resolve this bottleneck via two contributions:
1. **Taxonomy-Shaped MoE Routing Control ($J_{\text{tax}} + \mathcal{L}_{\text{temp}}$):** Supervising the internal sparse routers of a 4.9M-parameter, 3-layer, 8-expert Sparse Mixture-of-Experts (SMoE) Transformer ([Astro-MoE, Cádiz-Leyton et al., 2025](https://arxiv.org/abs/2507.12611)) with clip-stabilized Light-Curve-Level Taxonomy Mutual Information ($J_{\text{tax}}$, adapted from [Li et al., 2026](https://arxiv.org/abs/2610.07332)) and smooth-flux-gradient Temporal Consistency ($\mathcal{L}_{\text{temp}}$).
2. **MoE Routing Atypicality ($\mathcal{A}_{\text{route}}$):** A zero-extra-training inference statistic combining Cross-Layer Routing Path Divergence ($S_{\text{path}}$), Temporal Expert-Switch Rate ($S_{\text{switch}}$), and Blue-vs.-Red Passband Routing Asymmetry ($S_{\text{band}}$) prior to sequence pooling:
   $$\mathcal{A}_{\text{route}}(i) = \tilde{S}_{\text{path}}(i) + \alpha \tilde{S}_{\text{switch}}(i) + \beta \tilde{S}_{\text{band}}(i), \qquad (\alpha = 0.85,\; \beta = 0.90).$$

### Benchmark Results on LSST 6-Band (`u,g,r,i,z,Y`) ELAsTiCC/PLAsTiCC & FLARE

| Anomaly Scoring Method | Recall @ 1% FAR | AUROC | Gain vs. Output Entropy |
| :--- | :---: | :---: | :---: |
| **Composite $\mathcal{A}_{\text{route}}$ (Ours, 30 Epochs MPS)** | **98.0%** | **0.9993** | **+96.2 pp** |
| Component 3 Only: Band Asymmetry ($S_{\text{band}}$) | 98.5% | 0.9990 | +96.8 pp |
| Component 1 Only: Cross-Layer Path Divergence ($S_{\text{path}}$) | 84.5% | 0.9968 | +82.8 pp |
| Component 2 Only: Temporal Switch Rate ($S_{\text{switch}}$) | 62.0% | 0.9863 | +60.2 pp |
| Baseline 3: Penultimate Mahalanobis Distance | 9.9% | 0.6372 | +8.2 pp |
| Baseline 1: Output Predictive Entropy | 1.8% | 0.6708 | +0.0 pp |
| Baseline 2: Maximum Softmax Probability (MSP) | 0.3% | 0.6794 | -1.5 pp |

- **Expert Specialization:** Within-vs.-Cross Physical Family Cosine Similarity Gap increases from **`0.0000`** (Vanilla Shazeer Load-Balancing) to **`0.5280`** (`+4,872,899.6%`).
- **Per-Class Recall @ 1% FAR:** **100.0%** on Kilonovae (LSST `target=64`), **98.3%** on Intermediate Luminosity Optical Transients (ILOTs), **100.0%** on Pair-Instability Supernovae (PISN), and **90.0%** on FLARE rare transients (FBOT / Ca-rich Gap / LBV).

---

## Repository Structure

```text
├── s2_s11_astro_moe_routing_pilot.py         # Core 4.9M-param PyTorch (MPS/CUDA/CPU) Astro-MoE trainer & A_route evaluator
├── download_and_format_pilot_datasets.py     # Zenodo LSST (Record 2539456) & NERSC ELAsTiCC peak-SNR light-curve formatter
├── generate_paper1_figures_and_tables.py     # Generates Figures 1–4 (.pdf/.png) & syncs LaTeX booktabs tables in-place
├── s11_astro_moe_routing_arxiv_manuscript.tex # Full LaTeX manuscript (pdflatex / Overleaf / arXiv TeX Live ready)
├── pilot_data/
│   └── s11_live_gate_report.json             # 30-epoch MPS benchmark evaluation metrics & per-class OOD recall breakdown
└── publication_figures/
    ├── fig1_expert_specialization_heatmap.pdf
    ├── fig2_kilonova_vs_sn_routing_trajectory.pdf
    ├── fig3_low_far_roc_comparison.pdf
    ├── fig4_per_class_ood_recall_bars.pdf
    └── paper1_ablation_tables.tex
```

---

## Quickstart & Reproducibility

### 1. Environment Setup

Using [`uv`](https://github.com/astral-sh/uv) (or standard `pip`):

```bash
uv venv
source .venv/bin/activate
uv pip install numpy scipy pandas pyarrow matplotlib torch
```

### 2. Download & Format Peak-Centered LSST 6-Band Light Curves

Downloads the official LSST 6-band training photometry and metadata from [Zenodo Record 2539456](https://zenodo.org/records/2539456) and NERSC ELAsTiCC taxonomy tables, slices each light curve into a $T=32$ observation window centered around peak signal-to-noise ratio ($\tau_{\text{peak}} = \operatorname{argmax}_\tau |f_{i,\tau}|/\sigma_{i,\tau}$), and saves the stratified tensors:

```bash
uv run python download_and_format_pilot_datasets.py \
    --mode s2-download \
    --data-dir ./pilot_data \
    --n-id-curves 50000
```

*(For an instant offline self-test without downloading external files, run `--mode offline-verify`.)*

### 3. Train Vanilla vs. Taxonomy-Routed Astro-MoE & Evaluate $\mathcal{A}_{\text{route}}$

On Apple Silicon (`--device mps`), NVIDIA GPU (`--device cuda`), or CPU (`--device cpu`):

```bash
uv run python s2_s11_astro_moe_routing_pilot.py \
    --train-pytorch \
    --device mps \
    --epochs 30 \
    --dataset-npz ./pilot_data/elasticc_50k_flare_formatted.npz \
    --features-npz ./pilot_data/astro_moe_50k_mps_outputs.npz \
    --output ./pilot_data/s11_live_gate_report.json
```

### 4. Regenerate Publication Figures & Compile the LaTeX Manuscript

```bash
uv run python generate_paper1_figures_and_tables.py \
    --scorecard-json ./pilot_data/s11_live_gate_report.json \
    --features-npz ./pilot_data/astro_moe_50k_mps_outputs.npz \
    --tex-path s11_astro_moe_routing_arxiv_manuscript.tex \
    --out-dir ./publication_figures

pdflatex -interaction=nonstopmode s11_astro_moe_routing_arxiv_manuscript.tex
pdflatex -interaction=nonstopmode s11_astro_moe_routing_arxiv_manuscript.tex
```

---

## Citation

If you use this code, routing objective, or benchmark evaluation in your research, please cite:

```bibtex
@article{sinha2026astromoerouting,
  title   = {Taxonomy-Shaped Mixture-of-Experts Routing Atypicality for Zero-Shot Out-of-Taxonomy Transient Discovery in Rubin/LSST and ZTF},
  author  = {Sinha, Anurag},
  journal = {zenodo preprint},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).

# UTR-Diffusion

Official implementation of the paper:

> UTR-Diffusion: Conditional Diffusion Modeling for Multi-objective and Constrained UTR Design  
> ECCB 2026 Proceedings

UTR-Diffusion is a diffusion-based generative framework for 5′ UTR sequence design that enables:

- Continuous control of translation-related indicators:
  - Mean Ribosome Load (MRL)
  - Minimum Free Energy (MFE)
- Explicit codon- or amino-acid constraints at user-specified positions
- Multi-objective conditional generation

---

## 🔬 Overview

UTR-Diffusion supports:

- Unconditional generation
- Continuous conditional generation (MRL, MFE)
- Constrained generation (codon / amino acid clamping)
- Multi-label generation (MRL + MFE)
- Evaluation pipeline (MRL/MFE prediction)

---

## 📦 Installation (Conda)

We recommend installing dependencies via the provided `environment.yml`.

```bash
git clone https://github.com/satolab-isct/utr-diffusion
cd utr-diffusion

# create conda environment
conda env create -f environment.yml

# activate
conda activate utr-diffusion

## Quick Start (CLI)
1) Generate constrained sequences (amino-acid mode)

Example: generate sequences targeting MRL=8.0, MFE=-2.0 with amino-acid constraints at specific positions.

python design_utr.py \
  --mode amino \
  --targets "8.0,-2.0" \
  --amino 5:R 26:L \
  --out outputs/amino_demo.fasta \
  --device cuda:0

Output:

outputs/amino_demo.fasta

Optional: Evaluate generated sequences (MRL/MFE prediction)

If you want to evaluate MRL/MFE of generated sequences (i.e., produce a CSV report), you need an additional repository:

utr-diffusion-eval (evaluation pipeline)

Setup evaluation repository
git clone https://github.com/satolab-isct/utr-diffusion-eval ../UTR-Diffusion-eval
Generate + evaluate in one command
python design_utr.py \
  --mode amino \
  --targets "8.0,-2.0" \
  --amino 5:R 26:L \
  --out outputs/amino_demo.fasta \
  --do-eval \
  --eval-repo ../UTR-Diffusion-eval \
  --device cuda:0

Output:

outputs/amino_demo.fasta

outputs/amino_demo.csv ← contains predicted MRL/MFE for generated sequences

Note: --eval-repo should point to the local path of utr-diffusion-eval.

Arguments (Summary)

--mode: generation mode (amino, codon, or other supported modes)

--targets: "MRL,MFE" (two comma-separated continuous targets)

--amino: amino-acid constraints as pos:AA pairs (e.g., 5:R 26:L)

--out: output FASTA path

--device: cuda:0 or cpu

--do-eval: enable post-generation evaluation

--eval-repo: path to utr-diffusion-eval

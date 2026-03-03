# UTR-Diffusion

Official implementation of the paper:

> **UTR-Diffusion: Conditional Diffusion Modeling for Multi-objective and Constrained UTR Design**  
> ECCB 2026 Proceedings

UTR-Diffusion is a diffusion-based generative framework for 5′ UTR sequence design that enables:

- **Continuous control of translation-related indicators**
  - Mean Ribosome Load (MRL)
  - Minimum Free Energy (MFE)
- **Explicit codon- or amino-acid constraints** at user-specified positions
- **Multi-objective conditional generation**

---

## 🔬 Overview

UTR-Diffusion supports:

- Unconditional generation
- Continuous conditional generation (MRL, MFE)
- Constrained generation (codon / amino-acid clamping)
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
```
---

## 🚀 Quick Start (CLI)
### 1）Generate constrained sequences (amino-acid mode)

Example: generate sequences targeting MRL=8.0, MFE=-2.0 with amino-acid constraints at specific positions.

```bash
python design_utr.py \
  --mode amino \
  --targets "8.0,-2.0" \
  --amino 5:R 26:L \
  --out outputs/amino_demo.fasta \
  --device cuda:0
```

Output:

outputs/amino_demo.fasta

### 2）📊 Optional: Evaluate generated sequences (MRL/MFE prediction)

To predict MRL/MFE for generated sequences (i.e., produce a CSV report), you need an additional repository:

utr-diffusion-eval (evaluation pipeline)

Setup evaluation repository

```bash
git clone https://github.com/satolab-isct/utr-diffusion-eval ../UTR-Diffusion-eval
```

Generate + evaluate in one command

```bash
python design_utr.py \
  --mode amino \
  --targets "8.0,-2.0" \
  --amino 5:R 26:L \
  --out outputs/amino_demo.fasta \
  --do-eval \
  --eval-repo ../UTR-Diffusion-eval \
  --device cuda:0
```

Output

outputs/amino_demo.fasta

outputs/amino_demo.csv (predicted MRL/MFE values)

⚠️ --eval-repo must point to the local path of utr-diffusion-eval.

⚙️ Arguments (Summary)
Argument	Description

--mode	Generation mode (amino, codon, etc.)

--targets	"MRL,MFE" continuous targets

--amino	Amino-acid constraints (pos:AA, e.g., 5:R 26:L)

--out	Output FASTA file path

--device	cuda:0 or cpu

--do-eval	Enable post-generation evaluation

--eval-repo	Path to utr-diffusion-eval
